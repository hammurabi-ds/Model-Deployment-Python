import requests

payload = {"data":"buy tickets last minute fast buy now and get two for free"}
print(payload)
prediction = requests.get("http://0.0.0.0:9999/", json=payload, verify=False)
print(prediction.json())


