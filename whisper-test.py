import requests
import argparse
parser = argparse.ArgumentParser(description='Test API')
parser.add_argument('--audio_path', type=str)
audio_path = parser.parse_args().audio_path
data = {
    "audio_path": audio_path
}
url = f'http://127.0.0.1:9233/asr'
response = requests.post(url, data=data, timeout=3600)
#print(response.text)
print(response.json()["data"]["text"])