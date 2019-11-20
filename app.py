from flask import Flask
from flask_restful import Api
from resources.Predict import Predict
from resources.Front import Front

app = Flask(__name__, template_folder='templates')
api = Api(app)

api.add_resource(Front, '/')
api.add_resource(ServeModel, '/predict')

