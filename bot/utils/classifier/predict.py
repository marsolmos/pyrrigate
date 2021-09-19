'''Prediction Script'''
import cv2
from io import BytesIO
import json
import numpy as np
import os
import yaml

# Don't display general information messages from tf-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predict_plant(input_stream):
    '''
    Predict image species based on incoming image

    Args:
        image (io bytes): image received from telegram dispatcher

    Returns:
        (str): Predicted plant species name in latin
    '''
    # Read parameters and assign them to a local variable
    params = yaml.safe_load(open("params.yaml"))["train"]
    model_name = params["model_name"]

    # Load saved model
    model = tf.keras.models.load_model('utils/classifier/dvc_objects/model')
    print('TensorFlow trained model correctly loaded')

    # Load mapping dict for test classes
    map_file = "utils/classifier/dvc_objects/map_classes.json"
    with open(map_file) as json_file:
        map_classes = json.load(json_file)

    # Predict with the trained model
    image = cv2.imdecode(np.fromstring(input_stream.read(), np.uint8), 1) # Convert IO Bytes object into array
    image = cv2.resize(image, dsize=(150, 150), interpolation=cv2.INTER_CUBIC) # Resize image
    image = image / 255.0 # Normalize values
    image = image.reshape(-1, 150, 150, 3) # Reshape array into trained's model expected shape
    Y_pred = model.predict(image)
    y_pred = np.argmax(Y_pred, axis=1)
    # Map classes with class_dict
    y_pred = map_classes[str(y_pred[0])]

    return y_pred
