import pickle
import pandas as pd
import joblib
from flask_restful import Resource
from flask import request, jsonify
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE

class CheckInputSchema(Schema):
    """
    Use this class to check input format, check out marshmallow
    
    Example checking 
    payload = {"data":{"names": ["MailID","EmailText"],"ndarray": ["id9282","text here"]}}

    data = fields.Dict(keys=fields.Str(),
        values=fields.List(fields.Str(required=True, validate=validate.Length(min=1)),
        required=True,
        validate=validate.Length(min=2, max=2)))
    """

    class Meta:
        unknown = EXCLUDE

class ServeModel(Resource):
    def get(self):

        # Load preprocessor, model and check input 
        try:
            clf = pickle.load(open(os.getcwd() + "/models/model.pkl","rb"))
            json_ = request.json
            prep_pipeline = joblib.load(os.getcwd() + "/models/prep_pipeline.joblib")
            query = prep_pipeline.transform(list(json_.values()))
        except (FileNotFoundEror, ValidationError) as e:
            return jsonify({'mesage':str(e)})

        data = [' '.join(x) for x in query]
        prediction = list(clf.predict(data))
        return jsonify({'prediction': [str(x) for x in prediction]})

