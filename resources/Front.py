from flask_restfull import Resource
from flask import render_template, make_response

class Front(Resource):
    def get(self):
        return make_response(render_template("front.html"))

