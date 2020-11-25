import unittest
import requests

class RequestTest(unittest.TestCase):

    def test_predict(self):
        
        self.url = "http://0.0.0.0:9999/predict"
       
        payload = {
                "data":
                {
                "names": ["feature1", "feature2", "feature3"],
                "ndarray": ["string feature",3,"another number"]
                }
                }

        headers=requests.utils.default_headers()
        prediction = requests.get(self.url, json=payload, verify=False, headers=headers)
        print(prediction.json())
        #this checks that length of output is equal to one. change this to match your correct output
        self.assertEqual(1, len(prediction.json()))
        
if __name__ == '__main__':
    unittest.main()

