'''Evaluation Script'''
import json
import math
import numpy as np
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
confusion_file = "utils/classifier/dvc_objects/confusion.json"
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

# Create mapping dict for test classes
map_classes = {}
index = 0
for i in os.listdir(test_dir):
    map_classes[index] = i
    index += 1

# Note that the test data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow validation images in batches of 32 using val_datagen generator
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch,
        seed=seed,
        class_mode='categorical')

# Predict with the trained model
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes
# Map classes with class_dict
y_pred = [map_classes[k] for k in y_pred]
y_true = [map_classes[k] for k in y_true]

# Evaluate model metrics
results = model.evaluate(test_generator)
test_loss = results[0]
test_acc = results[1]
test_auc = results[2]
test_prc = results[3]
test_recall = results[4]
# Define True Positive Ratio and True Negative Ratio as
# TPR: Percentage of classes correctly classified
# TNR: Percentage of classes incorrectly classified
total = len(y_true)
tp = len([x for x, y in zip(y_true, y_pred) if x == y])
test_tpr = tp/total
test_tnr = 1-test_tpr

with open(scores_file, "w") as fd:
    json.dump({
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_prc": test_prc,
            "test_recall": test_recall,
            "test_tpr": test_tpr,
            "test_tnr": test_tnr
            }, fd, indent=4)

# Save Confusion Matrix
confusion_points = list(zip(y_true, y_pred))
with open(confusion_file, "w") as fd:
    json.dump(
        {
            "confusion": [
                {"actual": y_true, "predicted": y_pred}
                for y_true, y_pred in confusion_points
            ]
        },
        fd,
        indent=4,
    )
