from flask import Flask,jsonify,request
from classifer import getPrediction
app = Flask(__name__)
@app.route("/")
def hello_world():
    return "Hello i am Astitva"
@app.route("/predict-alphabate", methods=["POST"])
def predict_data():
    image=request.files.get("alphabate")
    prediction=getPrediction(image)
    return jsonify({
        "prediction":prediction
    }),200
if(__name__ == "__main__"):
    app.run(debug=True)