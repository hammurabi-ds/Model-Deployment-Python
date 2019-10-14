import requests

payload = {"data":"buy tickets last minute fast buy now and get two for free"}
print(payload)
headers=requests.utils.default_headers()

prediction = requests.get("http://0.0.0.0:9999/predict", json=payload, verify=False, headers=headers)
print(prediction.json())


