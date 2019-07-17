#!/usr/bin/env python3

from keras.applications import ResNet50, InceptionV3
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.applications import xception
from PIL import Image
import numpy as np
import flask
import io
from flask_cors import CORS

from InceptionV3Wrapper import InceptionV3Wrapper

app = flask.Flask(__name__)
CORS(app)
model = None

def load_InceptionV3():
    global model
    model = InceptionV3Wrapper()
    model.load_model()

# def load_ResNet50():
#     global model
#     model = InceptionV3Wrapper()
#     model.load_model()

@app.route("/test", methods=["GET"])
def testing():
    return(flask.jsonify("Test Get Request without"))

@app.route("/predict", methods=["POST"])
def return_predictions():
    predictions = {}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print("Received ", len(flask.request.files), " files. Processing...")
        # flask.request.files.items()
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = model.prepare_image(image)
            predictions = model.predict(image)

    # return the data dictionary as a JSON response
    response = flask.jsonify(predictions)
    response.headers.add('Access-Control-Allow-Origin', '*')	
    return response

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    # load_ResNet50()
    load_InceptionV3()
    app.run(host='0.0.0.0')

# Credits to : https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html
