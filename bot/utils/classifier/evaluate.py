'''Evaluation Script'''
import json
import math
import os
import pandas as pd
import pickle
import sys
import yaml

# Don't display general information messages from tf-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


scores_file = "utils/classifier/dvc_objects/test_scores.json"
prc_file = "utils/classifier/dvc_objects/prc.json"
roc_file = "utils/classifier/dvc_objects/roc.json"

# Read parameters and assign them to a local variable
params = yaml.safe_load(open("params.yaml"))["train"]
seed = params["seed"]
model_name = params["model_name"]
dataset = params["dataset"]
epochs = params["epochs"]
steps_per_epoch = params["steps_per_epoch"]
batch = params["batch"]
loss = params["loss"]
learning_rate = params["learning_rate"]

# Load saved model
model = tf.keras.models.load_model('utils/classifier/dvc_objects/model')

# Read parameters and assign them to a local variable
params = yaml.safe_load(open("params.yaml"))["train"]
dataset = params["dataset"]

# Define paths
base_dir = os.path.join("D:/Data Warehouse/plantabit", dataset)
test_dir = os.path.join(base_dir, 'test')

# Note that the test data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 32 using val_datagen generator
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch,
        seed=seed,
        class_mode='categorical')

# Evaluate model metrics
results = model.evaluate(test_generator, batch_size=batch)
test_loss = results[0]
test_acc = results[1]

with open(scores_file, "w") as fd:
    json.dump({"test_loss": test_loss, "test_acc": test_acc}, fd, indent=4)
