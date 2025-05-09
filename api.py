import os,time,sys
from pathlib import Path
import whisper
from huggingface_hub import snapshot_download
import librosa

root_dir=Path(__file__).parent.as_posix()
from flask_siwadoc import SiwaDoc
# ffmpeg
if sys.platform == 'win32':
    os.environ['PATH'] = root_dir + f';{root_dir}\\ffmpeg;' + os.environ['PATH']+f';{root_dir}/third_party/Matcha-TTS'
else:
    os.environ['PATH'] = root_dir + f':{root_dir}/ffmpeg:' + os.environ['PATH']
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':third_party/Matcha-TTS'
sys.path.append(f'{root_dir}/third_party/Matcha-TTS')
tmp_dir=Path(f'{root_dir}/tmp').as_posix()
logs_dir=Path(f'{root_dir}/logs').as_posix()
os.makedirs(tmp_dir,exist_ok=True)
os.makedirs(logs_dir,exist_ok=True)

from flask import Flask, request, render_template, jsonify,  send_from_directory,send_file,Response, stream_with_context,make_response,send_file
import logging
from logging.handlers import RotatingFileHandler
import subprocess
import shutil
import datetime
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav


import torchaudio,torch
from pathlib import Path
import base64
from IPython.display import Audio, display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 预加载ASR模型
#asr_model = whisper.load_model("large-v3")
asr_model = whisper.load_model("medium")
asr_model.to(device)


# 预加载SFT模型
tts_model = CosyVoice2('pretrained_models/CosyVoice2-0.5B',load_jit=False, load_trt=False,fp16=False,use_flow_cache=True)
#tts_model = None
# 懒加载clone模型，在第一次克隆时加载

# clone_model = CosyVoice('pretrained_models/CosyVoice-300M-Instruct')
# clone_model = None

'''
app logs
'''
# 配置日志
# 禁用 Werkzeug 默认的日志处理器
log = logging.getLogger('werkzeug')
log.handlers[:] = []
log.setLevel(logging.WARNING)

root_log = logging.getLogger()  # Flask的根日志记录器
root_log.handlers = []
root_log.setLevel(logging.WARNING)

app = Flask(__name__, 
    static_folder=root_dir+'/tmp', 
    static_url_path='/tmp')
siwa = SiwaDoc(app,title='CosyVoice API', description='CosyVoice API文档')
app.logger.setLevel(logging.WARNING) 


# 创建 RotatingFileHandler 对象，设置写入的文件路径和大小限制
file_handler = RotatingFileHandler(logs_dir+f'/{datetime.datetime.now().strftime("%Y%m%d")}.log', maxBytes=1024 * 1024, backupCount=5)
# 创建日志的格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 设置文件处理器的级别和格式
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(formatter)
# 将文件处理器添加到日志记录器中
app.logger.addHandler(file_handler)

max_val = 0.8

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    """
    对音频进行后处理，包括裁剪和归一化。

    :param speech: 输入的音频数据
    :param top_db: 裁剪时使用的分贝阈值
    :param hop_length: 跳跃长度
    :param win_length: 窗口长度
    :return: 处理后的音频数据
    """
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(tts_model.sample_rate * 0.2))], dim=1)
    return speech


def base64_to_wav(encoded_str, output_path):
    if not encoded_str:
        raise ValueError("Base64 encoded string is empty.")

    # 将base64编码的字符串解码为字节
    wav_bytes = base64.b64decode(encoded_str)

    # 检查输出路径是否存在，如果不存在则创建
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # 将解码后的字节写入文件
    with open(output_path, "wb") as wav_file:
        wav_file.write(wav_bytes)

    print(f"WAV file has been saved to {output_path}")


# 获取请求参数
def get_params(req):
    params={
        "text":"",
        "lang":"",
        "reference_audio":None,
        "reference_text":"",
        "speed":1.0,
        "stream":False,
        'role':""
    }
    # 原始字符串
    params['text'] = req.args.get("text","").strip() or req.form.get("text","").strip()
    
    # 速度
    params['speed'] = req.args.get("speed") or req.form.get("speed")
    params['speed'] = float(params['speed'])
    if params['speed'] < 0.5:
        params['speed'] = 0.5

    # 角色
    params['role'] = req.args.get("role","").strip() or req.form.get("role","").strip()

    # stream
    params['stream'] = req.args.get("stream") or req.form.get("stream")
    if params['stream'] == 'true' or params['stream'] == 'True':
        params['stream'] = True
    else:
        params['stream'] = False
        
    # 字符串语言代码
    params['lang'] = req.args.get("lang","").strip().lower() or req.form.get("lang","").strip().lower()
    # 兼容 ja语言代码
    if params['lang']=='ja':
        params['lang']='jp'
    elif params['lang'][:2] == 'zh':
        # 兼容 zh-cn zh-tw zh-hk
        params['lang']='zh'

    # 要克隆的音色文件    
    params['reference_audio'] = req.args.get("reference_audio",None) or req.form.get("reference_audio",None)
    encode=req.args.get('encode','') or req.form.get('encode','')
    if  encode=='base64':
        tmp_name=f'tmp/{time.time()}-clone-{len(params["reference_audio"])}.wav'
        base64_to_wav(params['reference_audio'],root_dir+'/'+tmp_name)
        params['reference_audio']=tmp_name
    # 音色文件对应文本
    params['reference_text'] = req.args.get("reference_text",'').strip() or req.form.get("reference_text",'')
    
    return params

@app.route('/close')
@siwa.doc()
def close_app():
    tmp_files=[]
    for f in os.listdir(tmp_dir):
        tmp_files.append(f"{tmp_dir}/{f}")
    del_tmp_files(tmp_files)
    print('正在关闭应用...')
    sys.exit(0)

def del_tmp_files(tmp_files: list):
    print('正在删除缓存文件...')
    for f in tmp_files:
        if os.path.exists(f):
            print('删除缓存文件:', f)
            os.remove(f)


# 实际批量合成完毕后连接为一个文件
def batch(tts_type,outname,params):
    global sft_model,tts_model
    if not shutil.which("ffmpeg"):
        raise Exception('必须安装 ffmpeg')    
    prompt_speech_16k=None

    if params['reference_audio']:
        pass
    else:
        ref = f"reference/{params['role']}"
        for f in os.listdir(ref):
            if f.endswith('.wav') or f.endswith('.mp3'):
                params['reference_audio'] = f"{ref}/{f}"
                break
    if params['reference_audio']:
        print(f"reference_audio: {params['reference_audio']}")
    else:
        raise Exception('未传入参考音频或未找到参考音频')
    
    if params['reference_text']:
        print(f"reference_text: {params['reference_text']}")
    if params['text']:
        print(f"text: {params['text']}")
    if tts_type!='tts':
        if not params['reference_audio'] or not os.path.exists(f"{root_dir}/{params['reference_audio']}"):
            raise Exception(f'参考音频未传入或不存在 {params["reference_audio"]}')
        ref_audio=f"{tmp_dir}/-refaudio-{time.time()}.wav" 
        try:
            subprocess.run(["ffmpeg","-hide_banner", "-ignore_unknown","-y","-i",params['reference_audio'],"-ar","16000",ref_audio],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   encoding="utf-8",
                   check=True,
                   text=True,
                   creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            raise Exception(f'处理参考音频失败:{e}')
        
        prompt_speech_16k = load_wav(ref_audio, 16000)
        # prompt_speech_16k = load_wav(params['reference_audio'], 16000)
    text=params['text']
    audio_list=[]
    stream=params['stream']
    flow = stream
    out_dir = f"output/{params['role']}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir,exist_ok=True)
    out_dir = f"{out_dir}/{outname}"
    if tts_type=='clone_eq' and params.get('reference_text'):
        print("params:",params)
        print()
        if tts_model is None:
            tts_model=CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False,use_flow_cache= True)
        for i, j in enumerate(tts_model.inference_zero_shot(text,params.get('reference_text'),prompt_speech_16k, stream=stream,speed=params['speed'])):
            audio_list.append(j['tts_speech'])

    elif tts_type=='clone' and params.get('reference_text'):
        if tts_model is None:
            tts_model=CosyVoice2('pretrained_models/CosyVoice2-0.5B',load_jit=False, load_trt=False,fp16=False,use_flow_cache=True)

        for i, j in enumerate(tts_model.inference_cross_lingual(text,prompt_speech_16k, stream=stream,speed=params['speed'])):
            audio_list.append(j['tts_speech'])
    else:
        if tts_model is None:
            tts_model=CosyVoice2('pretrained_models/CosyVoice2-0.5B',load_jit=False, load_trt=False,fp16=False,use_flow_cache=True)
        for i, j in enumerate(tts_model.inference_instruct2(text, stream=stream,speed=params['speed'])):
            audio_list.append(j['tts_speech'])
            display(Audio(j['tts_speech'].numpy(), rate=tts_model.sample_rate))

    audio_data = torch.concat(audio_list, dim=1)
    
    torchaudio.save(out_dir,audio_data, 24000, format="wav")   
    
    print(f"音频文件生成成功：{out_dir}")
    return out_dir



# 跨语言文字合成语音      
@app.route('/clone_mul', methods=['GET', 'POST'])        
@app.route('/clone', methods=['GET', 'POST'])   
@siwa.doc()     
def clone():
    '''
    跨语言文字合成语音
    请求方法: POST,GET
    请求参数:
        - text: 要合成的文本
        - lang: 文本语言
        - reference_audio: 要克隆的音色文件
        - reference_text: 音色文件对应文本
        - speed: 语速
        - encode: 参考音频编码方式,base64
        - response_type: 返回类型,stream或file
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    print('clone...')
    try:
        params=get_params(request)
        print(f"params: {params}")
        if not params['text']:
            return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
            
        outname=f"clone-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        outname=batch(tts_type='clone',outname=outname,params=params)
    except Exception as e:
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        return send_file(outname, mimetype='audio/x-wav')
    
@app.route('/clone_eq', methods=['GET', 'POST'])         
def clone_eq():
    '''
    同语言克隆
    请求方法: POST,GET
    请求参数:
        - text: 要合成的文本
        - lang: 文本语言
        - reference_audio: 要克隆的音色文件
        - reference_text: 音色文件对应文本
        - speed: 语速
        - encode: 参考音频编码方式,base64
        - response_type: 返回类型,stream或file
        - stream: 是否流式返回
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    print('clone_eq...')
    try:
        params=get_params(request)
        print(f"params: {params}")
        if not params['text']:
            return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
        if not params['reference_text']:
            return make_response(jsonify({"code":6,"msg":'同语言克隆必须传递引用文本'}), 500)  # 设置状态码为500
            
        outname=f"clone-{params['role']}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        outname=batch(tts_type='clone_eq',outname=outname,params=params)
    except Exception as e:
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        return send_file(outname, mimetype='audio/x-wav')
     
@app.route('/audio_list/<role>', methods=['GET', 'POST'])
def audio_list(role):
    '''
    获取音频列表
    请求方法: POST,GET
    请求参数:
        - role: 角色
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    print('audio_list...')
    # params = {
    #     "role": role
    # }
    audio_list=[]
    for f in os.listdir(f"output/{role}"):
        if f.endswith('.wav'):
            audio_list.append(f)
    return jsonify({"code":0,"msg":'success',"data":audio_list})

@app.route('/upload', methods=['POST'])
def upload():
    '''
    上传参考音频
    请求方法: POST
    请求参数:
        - role: 角色
        - file: 文件
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    print('upload...')
    role = request.form.get('role')
    if not role:
        return jsonify({"code": 6, "msg": '缺少角色参数'})
    if 'file' not in request.files:
        return jsonify({"code": 6, "msg": '缺少文件参数'})
    file = request.files['file']
    upload_dir = f"reference/{role}"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    file.save(file_path)
    return jsonify({"code": 0, "msg": 'success', "data": file_path})

@app.route('/asr', methods=['POST'])
@siwa.doc()
def asr():
    '''
    语音识别
    请求方法: POST
    请求参数:
        - audio_path: 音频文件路径
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    global asr_model
    print('asr...')
    audio_path = request.form.get('audio_path')
    if not audio_path:
        return jsonify({"code": 6, "msg": '缺少音频文件路径参数'})
    if not os.path.exists(audio_path):
        return jsonify({"code": 6, "msg": '音频文件不存在'})

    audio = whisper.load_audio(audio_path)
    # audio = whisper.pad_or_trim(audio) # 填充或裁剪音频
    result = asr_model.transcribe(audio,initial_prompt="生于忧患，死于安乐。不亦快哉！")
    return jsonify({"code": 0, "msg":'success', "data": result})

@app.route('/stream', methods=['POST'])
@siwa.doc()
def stream():
    '''
    流式合成
    请求方法: POST
    请求参数:
        - text: 要合成的文本
        - lang: 文本语言
        - reference_audio: 要克隆的音色文件
        - reference_text: 音色文件对应文本
        - speed: 语速
        - encode: 参考音频编码方式,base64
        - response_type: 返回类型,stream或file
    返回:
        - code: 状态码
        - msg: 消息
        - data: 数据
    '''
    print('stream...')
    try:
        params=get_params(request)
        print(f"params: {params}")
        if not params['text']:
            return make_response(jsonify({"code":6,"msg":'缺少待合成的文本'}), 500)  # 设置状态码为500
        outname=f"stream-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S-')}.wav"
        outname=batch(tts_type='stream',outname=outname,params=params)  
    except Exception as e:
        return make_response(jsonify({"code":8,"msg":str(e)}), 500)  # 设置状态码为500
    else:
        def generate():
            with open(outname, 'rb') as f:
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        break
                    yield chunk
        return Response(generate(), mimetype='audio/x-wav')
     


if __name__=='__main__':
    host='127.0.0.1'
    port=9233
    print(f'\n启动api:http://{host}:{port}\n')
    # try:
    #     from waitress import serve
    # except Exception:
    #     print('未安装waitress,使用flask启动')
    #     app.run(port=port)
    # else:
    #     print('使用waitress启动')
    #     serve(app,port=port)                                                                                                                                                                                                                        
    app.run(port=port)
    