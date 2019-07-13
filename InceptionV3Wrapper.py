from keras.applications import InceptionV3, xception
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np

class InceptionV3Wrapper:
  def __init__(self):
    self.model = None

  def load_model(self):
    self.model = InceptionV3()
    self.model._make_predict_function()

  # TODO : wrap common logic in abstract class
  def prepare_image(self, image):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = xception.preprocess_input(image)

    # return the processed image
    return image

  def predict(self, image):
    ## TODO : turn to warning and load the model at this time
    if self.model is None:
      raise Exception('Must call load_model() before infering')
    predictions = self.model.predict(image)
    results = imagenet_utils.decode_predictions(predictions)
    
    json_predictions = {"success": False}
    json_predictions["predictions"] = []
    # loop over the results and add them to the list of
    # returned predictions
    for (imagenetID, label, prob) in results[0]:
        r = {"label": label, "probability": round(float(prob),2)}
        json_predictions["predictions"].append(r)
    json_predictions["success"] = True
    return json_predictions

# For ResNet50
# from keras.applications import imagenet_utils
# image = imagenet_utils.preprocess_input(image)    