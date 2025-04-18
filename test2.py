import requests
import argparse
parser = argparse.ArgumentParser(description='Test API')
parser.add_argument('--text', type=str, default='本党本军所到之处，民众竭诚欢迎，真可谓占尽天时，那种勃勃生机、万物竞发的境界，犹在眼前。短短二十年之后，这里竟 至于一变而成为我们的葬身之地了么？无论怎么样，会战兵力是以八十万对六十万，优势在我！')
parser.add_argument('--role', type=str, default='huang')
#parser.add_argument('--audio_path', type=str, default='./reference/huang/1.wav')
parser.add_argument('--audio_path', type=str, default='')
parser.add_argument('--reference_text', type=str, default='你好吗？今天天气不错，你的心情怎么样？')
parser.add_argument('--speed', type=float, default=1.0)
parser.add_argument('--stream', type=bool, default=False)
url = 'http://127.0.0.1:9233'
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
response=requests.post(f'{url}/clone_eq',data=data,timeout=3600)
if response.status_code==200:
    print("success")
else:
    print(response.text)