from flask import Flask, request, jsonify, render_template
from flask_restful import Resource, Api
import os
import pickle
import pandas as pd
import joblib

app = Flask(__name__)
api = Api(app)

class ServeModel(Resource):
    def get(self):
        clf = pickle.load(open(os.getcwd() + "/models/model.pkl","rb"))
        json_ = request.json
        prep_pipeline = joblib.load(os.getcwd() + "/models/prep_pipeline.joblib")
        query = prep_pipeline.transform(list(json_.values()))
        
        data = [' '.join(x) for x in query]
        prediction = list(clf.predict(data))
        return jsonify({'prediction': [str(x) for x in prediction]})

class Front(Resource):
    def get(self):
        return "<h1 style='color:blue'>Serving Machine learning models!</h1>" 

api.add_resource(Front, '/')
api.add_resource(ServeModel, '/predict')

