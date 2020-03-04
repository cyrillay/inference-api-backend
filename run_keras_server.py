# import the necessary packages
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from tensorflow.python.keras.backend import set_session
from keras.backend import get_session
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
from flask_cors import CORS

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)
model = None
graph = None
session = None


def load_model():
    global model
    global graph
    global session
    session = get_session()
    init = tf.global_variables_initializer()
    session.run(init)
    graph = tf.get_default_graph()
    model = ResNet50(weights="imagenet")
    model._make_predict_function()


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    return image


@app.route("/test", methods=["GET"])
def testing():
    return flask.jsonify("Hello !")


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print(flask.request.files)
        print(len(flask.request.files))
        for k, v in flask.request.files.items():
            print(k, v)
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            processed_image = prepare_image(image, target=(224, 224))
            print(f"Image size : {processed_image.shape}")
            global graph
            global session
            preds = None
            with graph.as_default():
                set_session(session)
                preds = model.predict(processed_image)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    response = flask.jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print("Loading Keras model starting Flask server...")
    load_model()
    app.run(host='0.0.0.0')
