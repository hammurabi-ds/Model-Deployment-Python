from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import os
import pickle
import pandas as pd
import joblib
00
app = Flask(__name__)
api = Api(app)

class ServeModel(Resource):
    def get(self):
        print("loading model")
        clf = pickle.load(open(os.getcwd() + "/models/model.pkl","rb"))
        json_ = request.json
        prep_pipeline = joblib.load(os.getcwd() + "/models/prep_pipeline.joblib")
        query = prep_pipeline.transform(list(json_.values()))
        
        joined=[]
        for i in range(len(query)):
            joined.append(' '.join(query[i]))

        print("making prediction")
        prediction = list(clf.predict(joined))
        return jsonify({'prediction': [int(x) for x in prediction]})
    
api.add_resource(ServeModel, '/')

