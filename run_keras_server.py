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

app = flask.Flask(__name__)
CORS(app)
session = None
graph = None
model = None


def load_model():
    global model
    global graph
    global session
    session = get_session()
    init = tf.global_variables_initializer()
    session.run(init)
    graph = tf.get_default_graph()
    model = ResNet50(weights="imagenet")
    # https://github.com/keras-team/keras/issues/6124
    model._make_predict_function()


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    print(f"Converted image to shape : {image.shape}")
    return image


@app.route("/test", methods=["GET"])
def testing():
    return flask.jsonify("Hello !")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Returns a JSON response containing the classification labels predicted by the neural network for the posted image
    files

    Usage : `curl -X POST -F image=@YOUR_IMAGE.JPG http://0.0.0.0:5000/predict`
    """
    data = {"success": False, "predictions": []}
    if flask.request.method == "POST":
        print(f"Received {len(flask.request.files)} files.")
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            processed_image = prepare_image(image, target=(224, 224))
            with graph.as_default():
                set_session(session)
                preds = model.predict(processed_image)
            results = imagenet_utils.decode_predictions(preds)
            for (_, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)
            data["success"] = True
    response = flask.jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return flask.jsonify(data)


if __name__ == "__main__":
    print("Loading Keras model starting Flask server...")
    load_model()
    app.run(host="0.0.0.0")
