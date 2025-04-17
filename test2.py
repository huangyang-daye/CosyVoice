import requests
import argparse
parser = argparse.ArgumentParser(description='Test API')
parser.add_argument('--text', type=str, default='这是一段测试文本，文本本身并无意义，请忽略，谢谢。在当今数字化时代，随着人们对电脑和互联网的依赖越来越深入，个人隐私的保护变得尤为重要。随着隐私窥探技术的不断发展，许多人担心自己的电脑会被黑客或其他人窥探。因此，许多用户正在寻找防窥屏软件，特别是那些基于Windows Hello的软件。在本文中，我们将介绍一些Windows的防窥屏软件，以帮助您保护您的个人隐私。')
parser.add_argument('--role', type=str, default='huang')
parser.add_argument('--audio_path', type=str, default='./reference/huang/1.wav')
parser.add_argument('--reference_text', type=str, default='你好吗？今天天气不错，你的心情怎么样？')
parser.add_argument('--speed', type=float, default=1.0)
parser.add_argument('--stream', type=bool, default=False)

args = parser.parse_args()
data={
    "text":args.text,
    "reference_audio":args.audio_path,
    "reference_text":args.reference_text,
    "speed":args.speed,
    "stream":args.stream,
    "role":args.role
}
print(data)
response=requests.post(f'http://127.0.0.1:9233/clone_eq',data=data,timeout=3600)
if response.status_code==200:
    print("success")
else:
    print(response.text)