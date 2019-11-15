from flask import Flask
from flask_restful import Api
from resources.Predict import Predict
from resources.Front import Front

app = Flask(__name__)
api = Api(app)

api.add_resource(Front, '/')
api.add_resource(ServeModel, '/predict')

