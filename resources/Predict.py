from flask_restful import Resource
from flask import request, jsonify

class ServeModel(Resource):
    def get(self):

        # Load preprocessor, model and check input 
        try:
            json_ = request.json
            query = list(json_.values())
        except (FileNotFoundEror) as e:
            return jsonify({'mesage':str(e)})

        prediction = list(query)
        return jsonify({'your input data was': prediction})

