from flask_restfull import Resource

class Front(Resource):
    def get(self):
        return "<h1 style='color:blue'>Serving Machine learning models!</h1>"

