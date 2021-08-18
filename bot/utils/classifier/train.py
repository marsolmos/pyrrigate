'''Preprocessing and training pipelines for Planta-Bit'''
import datetime
import json
import os
import pickle
import yaml
import zipfile
import matplotlib.pyplot as plt

# Don't display general information messages from tf-gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
l2_reg = params["l2_reg"]
momentum = params["momentum"]

# Define paths
base_dir = os.path.join("D:/Data Warehouse/plantabit", dataset)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_scores = "utils/classifier/dvc_objects/train_scores.json"
val_scores = "utils/classifier/dvc_objects/val_scores.json"

# Define GPU usage for training
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Initialize TensorBoard
model_log_name = model_name + "-" + dataset + "-" + datetime.datetime.now().strftime("%Y%m%d/%H%M%S")
log_dir = "utils/classifier/models/logs/" + model_log_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tensorboard = TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=True,
    update_freq='epochs', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, and zoom_range to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    )

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(227, 227),  # All images will be resized to 150x150
        batch_size=batch,
        seed=seed,
        class_mode='categorical')


# Flow validation images in batches of 32 using val_datagen generator
validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(227, 227),
        batch_size=batch,
        seed=seed,
        class_mode='categorical')

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = (227, 227, 3)

# First convolution extracts 96 filters that are 11x11
# Convolution is followed by a layer Normalization and
# max-pooling layer with a 2x2 window
model = Sequential()
model.add(layers.Conv2D(
                    96, kernel_size=(11, 11),
                    activation='relu', strides=(4, 4),
                    padding="valid", input_shape=img_input,
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Second convolution extracts 256 filters that are 5x5
# Convolution is followed by max-pooling layer with a 1x1 window
model.add(layers.Conv2D(
                    256, kernel_size=(5, 5),
                    activation='relu', strides=(1, 1),
                    padding="same",
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Third convolution extracts 384 filters that are 3x3
model.add(layers.Conv2D(
                    384, kernel_size=(3, 3),
                    activation='relu', strides=(1, 1),
                    padding="same",
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))

# Fourth convolution extracts 192 filters that are 3x3
model.add(layers.Conv2D(
                    384, kernel_size=(3, 3),
                    activation='relu', strides=(1, 1),
                    padding="same",
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))

# Fifth convolution extracts 192 filters that are 3x3
model.add(layers.Conv2D(
                    256, kernel_size=(3, 3),
                    activation='relu', strides=(1, 1),
                    padding="same",
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten feature map to a 1-dim tensor
model.add(layers.Flatten())

# Create a fully connected layer with ReLU activation and 4096 hidden units
model.add(layers.Dense(
                    4096, activation='relu',
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))

# Create a fully connected layer with ReLU activation and 4096 hidden units
model.add(layers.Dense(
                    4096, activation='relu',
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))

# Create output layer with a single node and sigmoid activation
model.add(layers.Dense(
                    12, activation='softmax',
                    kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)
                    ))

# Configure and compile the model
model.compile(loss=loss,
              optimizer=SGD(
                            learning_rate=learning_rate,
                            momentum=momentum
                            ),
              metrics=[
                    metrics.CategoricalAccuracy(),
                    metrics.AUC(),
                    metrics.Precision(),
                    metrics.Recall(),
              ])

# Display model summary
model.summary()

# Train model to train-validation data
history = model.fit(
      train_generator,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=1,
      verbose=1,
      callbacks=[tensorboard_callback]
      )

# Save the model to disk into historical archive folder
print('\nSaving model into historical registry as {}'.format(model_name))
save_name = 'utils/classifier/models/{}'.format(model_name)
model.save(save_name)

# Save model to disk for DVC - MLOps tracking
print('Saving model for DVC - MLOps tracking\n')
save_name = 'utils/classifier/dvc_objects/model'
model.save(save_name)

# Save metrics to disk for DVC - MLOps tracking
train_loss = history.history['loss'][-1]
train_acc = history.history['categorical_accuracy'][-1]
val_loss = history.history['val_loss'][-1]
val_acc = history.history['val_categorical_accuracy'][-1]
with open(train_scores, "w") as fd:
    json.dump({"train_loss": train_loss, "train_acc": train_acc}, fd, indent=4)
with open(val_scores, "w") as fd:
    json.dump({"val_loss": val_loss, "val_acc": val_acc}, fd, indent=4)
