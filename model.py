import os
import plaidml.keras
plaidml.keras.install_backend()
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten, MaxPool2D
import numpy as np
import pandas as pd
from warnings import filterwarnings
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# filter training warnings.
filterwarnings('ignore')
# pneumonia xray image path
image_path = r'chest_xray'
train_path = image_path + r'\train'
test_path = image_path + r'\test'
# image target shape
img_shape = (224,224,3)
# image generator
img_gen = ImageDataGenerator(
    zoom_range = 0.15,
    rescale = 1/255
)
# create VGG-7 model
model = Sequential()
model.add(Conv2D(8, (3,3), activation = 'relu', input_shape = img_shape))
model.add(Conv2D(8, (3,3), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(16, (3,3), activation = 'relu'))
model.add(Conv2D(16, (3,3), activation = 'relu'))
model.add(MaxPool2D())
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#create image batches
batch = 4
train_gen = img_gen.flow_from_directory(train_path,
                            target_size = img_shape[:2],
                            color_mode='rgb',
                            batch_size=batch,
                            class_mode='binary'
                            )
test_gen = img_gen.flow_from_directory(test_path,
                            target_size = img_shape[:2],
                            color_mode='rgb',
                            batch_size=batch,
                            class_mode='binary'
                            )
early_stop = EarlyStopping(monitor = 'val_accuracy', patience = 1)
model.fit_generator(train_gen, epochs = 30, validation_data = test_gen, callbacks = [early_stop])