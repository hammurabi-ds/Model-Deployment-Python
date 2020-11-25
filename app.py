from flask import Flask
from flask_restful import Api
from resources.Predict import ServeModel

app = Flask(__name__, template_folder='templates')
api = Api(app)

api.add_resource(ServeModel, '/predict')

