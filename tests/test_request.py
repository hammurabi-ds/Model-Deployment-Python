import unittest
import requests

class RequestTest(unittest.TestCase):

    def test_predict(self):
        
        self.url = "http://0.0.0.0:9999/predict"
       
        payload = {
                "data":
                {
                "names": ["MailID","EmailText"],
                "ndarray": ["id9282","text here"]
                }
                }

        headers=requests.utils.default_headers()
        prediction = requests.get(self.url, json=payload, verify=False, headers=headers)
        print(prediction.json())
        self.assertEqual(2, len(prediction.json()))
        
if __name__ == '__main__':
    unittest.main()

