#flask act as api between html and cnn model.
#flask app receives rsponse from html front end, receives image
# and send response in json format. 
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.predict import PredictionPipeline #prediction pipeline imported from predict.py

#These environment variables is for stability of text data, may not directly related to cnn model
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__) #instance of the Flask web application is created
CORS(app) #cors is used to enable cross-origin resource sharing.

#clienaapp class will take image from html and predict(calls pred.Pipeline class from predict.py) 
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

#this code is defining a route that responds to HTTP GET requests at the root URL and serves an 
# HTML page ('index.html') to the user, effectively providing the main landing page for the 
# web application
@app.route("/", methods=['GET'])
@cross_origin() #allows cross-origin requests to this route.
def home():
    return render_template('index.html')

'''
@app.route("/train", methods=['GET','POST'])
@cross_origin() #allows cross-origin requests to this route.
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"
'''

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == "__main__":
    clApp = ClientApp()
    #app.run(host='0.0.0.0', port=8080) #local host
    # app.run(host='0.0.0.0', port=8080) #for AWS
    app.run(host='0.0.0.0', port=80) #for AZURE