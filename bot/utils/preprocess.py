'''Preprocessing pipelines for Planta-Bit'''
import os
import zipfile
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


MODEL_NAME = 'Plantabit-alpha-0-1'
N_AUGMENTED_IMAGES = 10 # Number of augmented images created with each real image
seed = 123 # Random number generator seed to be used along preprocessing
base_dir = "D:/Data Warehouse/plantabit/0_rawdata"
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training pictures
train_front_dir = os.path.join(train_dir, 'front')
train_rear_dir = os.path.join(train_dir, 'rear')
train_front_left_dir = os.path.join(train_dir, 'front_left')
train_front_right_dir = os.path.join(train_dir, 'front_right')
train_left_dir = os.path.join(train_dir, 'left')
train_right_dir = os.path.join(train_dir, 'right')
train_rear_left_dir = os.path.join(train_dir, 'rear_left')
train_rear_right_dir = os.path.join(train_dir, 'rear_right')
train_other_dir = os.path.join(train_dir, 'other')

# Directory with our validation pictures
validation_front_dir = os.path.join(validation_dir, 'front')
validation_rear_dir = os.path.join(validation_dir, 'rear')
validation_front_left_dir = os.path.join(validation_dir, 'front_left')
validation_front_right_dir = os.path.join(validation_dir, 'front_right')
validation_left_dir = os.path.join(validation_dir, 'left')
validation_right_dir = os.path.join(validation_dir, 'right')
validation_rear_left_dir = os.path.join(validation_dir, 'rear_left')
validation_rear_right_dir = os.path.join(validation_dir, 'rear_right')
validation_other_dir = os.path.join(validation_dir, 'other')


# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    # horizontal_flip=True,
    )

# Note that the validation data should not be augmented!
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        validation_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=1,
        save_to_dir="D:/Data Warehouse/plantabit/3_extended/train",
        seed=seed,
        class_mode='categorical')

print(train_generator.class_indices)
for _ in range(len(train_generator)):
    for _ in range(N_AUGMENTED_IMAGES):
        train_generator.next()

# # Flow validation images in batches of 32 using val_datagen generator
# validation_generator = val_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='categorical')
