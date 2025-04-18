import requests
import argparse
parser = argparse.ArgumentParser(description='Test API')
parser.add_argument('--role', type=str, default='huang')
role = parser.parse_args().role
url = f'http://127.0.0.1:9233/audio_list/{role}'
response = requests.get(url)
print(response.text)
print(response.json()['data'])