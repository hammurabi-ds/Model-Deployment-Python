import requests

payload = {"data":"raw text example"}
print(payload)
headers=requests.utils.default_headers()

prediction = requests.get("http://0.0.0.0:PORT/predict", json=payload, verify=False, headers=headers)
print(prediction.json())


