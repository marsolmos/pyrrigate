'''Prediction Script'''
import numpy as np
import os
import yaml

# Don't display general information messages from tf-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def predict_plant(image):
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

    # Create mapping dict for test classes
    map_classes = {}
    index = 0
    for i in os.listdir(test_dir):
        map_classes[index] = i
        index += 1

    # Predict with the trained model
    image = image / 255.0
    print('Image looks like')
    print(image)
    Y_pred = model.predict(image)
    y_pred = np.argmax(Y_pred, axis=1)
    # Map classes with class_dict
    y_pred = [map_classes[k] for k in y_pred]

    return y_pred
