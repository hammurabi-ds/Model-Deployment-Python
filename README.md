# Model-Deployment-RESTful
model deployment in flask restful

## Usage

1. Dump your model in pickle format under `models`
2. Define how the predict endpoint should do. (extract payload data from the json and put it in dataframe for preprocessor & model predict call)
3. define library dependencies in `requirements.txt`


## The predict class

Prepare the endpoint to load your model, preprocess your data and inport the right dependencies

```python
from flask_restful import Resource
from flask import request, jsonify

class ServeModel(Resource):
    def get(self):
        try:
            json_ = request.json
            query = list(json_.values())

            #Here load model from pickel format

            #call the predict function
        except (FileNotFoundEror) as e:
            return jsonify({'mesage':str(e)})

        # put prediction as list and return json
        prediction = list(query)
        return jsonify({'your input data was': prediction})
```

## Steps for deployment:

1. Build the docker container by `docker build . -t APP_NAME`
2. Run the docker container by `docker run -d -p PORT --name CONTAINER_NAME APP_NAME`
3. Send a test http request using `test_request.py`. Remember to modify the request file. 
