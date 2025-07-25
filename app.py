import requests

url = "http://127.0.0.1:8000/"
data = {"text": "오늘 날씨가 흐려요"}

response = requests.post(url, json=data)
print(response.json())