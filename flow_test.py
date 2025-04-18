import sys
import os
sys.path.append(os.path.abspath('third_party/Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
from IPython.display import Audio, display

cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=True,
    load_trt=False
)

text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
style_instruction = '用四川话说这句话'

prompt_speech_16k = load_wav('.\reference\huang\1.wav', 16000)


print("开始流式生成音频...")

for i, result in enumerate(cosyvoice.inference_instruct2(text, style_instruction, prompt_speech_16k, stream=True)):

    audio_data = result['tts_speech']
    audio_np = audio_data.numpy()

    print(f"播放第 {i + 1} 段音频：")
    display(Audio(audio_np, rate=cosyvoice.sample_rate))

print("音频流式输出完成。")