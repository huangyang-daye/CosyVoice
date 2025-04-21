import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import whisper

# import os
# path = "reference/huang/1.wav"
# print(os.path.exists(path))  # 检查文件是否存在



cosyvoice = CosyVoice2('D:\WorkSpace\CosyVoice\pretrained_models\pretrained_models\CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True,use_flow_cache=True)
print("CosyVoice2 loaded")
prompt_speech_16k = load_wav(r'reference\huang\1_fixed.wav', 16000)
print("loaded wav")
prompt_text = whisper.load_model("medium").transcribe(r"reference\huang\1_fixed.wav")['text']
print(prompt_text)
for i, j in enumerate(cosyvoice.inference_zero_shot('屈原（约公元前340—前278年），战国时期楚国诗人、政治家，中国浪漫主义文学奠基人。他早年深受楚怀王信任，主张举贤修法、联齐抗秦，因遭贵族排挤被流放‌。在秦军攻破楚国都城后，屈原悲愤投江，以死明志，其爱国精神成为中华民族的精神丰碑‌。他开创"楚辞"文体，《离骚》《天问》等作品以瑰丽想象和爱国情怀传世，其中"路漫漫其修远兮，吾将上下而求索"等名句彰显其高洁品格‌。端午节赛龙舟、食粽子等习俗即源于对他的纪念‌。',
                                                    prompt_speech_16k=prompt_speech_16k,
                                                    prompt_text=prompt_text,
                                                    stream=True)): 
    continue