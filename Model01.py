## Imports

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers

## Data Generators for train/test/val

train_data_generator = ImageDataGenerator(
    horizontal_flip = True,
    rotation_range = 10,
    brightness_range = [-0.5 , 0.5]

)

test_data_generator = ImageDataGenerator(

)

validation_data_generator = ImageDataGenerator(

)


## File import

train_data = train_data_generator.flow_from_directory(
    "data/training",
    target_size = (150, 150),
    batch_size = 176,
    class_mode = "binary",
    color_mode = "grayscale",
    shuffle = True,
    seed=1984)

test_data = test_data_generator.flow_from_directory(
    "data/testing",
    target_size = (150, 150),
    batch_size = 22,
    class_mode = "binary",
    color_mode = "grayscale",
    shuffle = True,
    seed = 1963
    )

validation_data = validation_data_generator.flow_from_directory(
    "data/validation",
    target_size = (150, 150),
    batch_size = 22,
    class_mode = "binary",
    color_mode = "grayscale",
    shuffle = True,
    seed = 1989
)


## Task 3 Constructing the Network

model = models.Sequential()

# Input Layer
model.add(layers.Conv2D(
    32,
    (3, 3),
    activation="relu",
    input_shape=(150, 150, 1),
    data_format="channels_last"
))

# Feature Extraction
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizers.l1(l1=0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu", bias_regularizer=regularizers.l2(1e-4)))


# Classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

# Compile Model

model.compile(
    optimizer='Adam',
    loss='BinaryCrossentropy',
    metrics=['accuracy'])

## Fit Model

model_history = model.fit(
    train_data,
    steps_per_epoch = 10,
    epochs = 10,
    validation_data = validation_data,
    validation_steps = 10
)

model.evaluate(test_data)

## Save Model

model.save("model01.h5")

## Evaluate Current model

model.evaluate(test_data)

## Evaluate Load Model

#model_load = keras.models.load_model('j_model.h5')

#model_load.evaluate(test_data)

#model_load.evaluate(validation_data)

