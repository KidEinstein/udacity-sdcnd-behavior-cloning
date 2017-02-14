import pandas as pd
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D
from keras.layers import Lambda
from keras.layers.core import Dropout
from keras.optimizers import Adam

# Folder with the training data
simulator_output_folder = "training_data_final"

# Reading driving log into a pandas data frame for processing
driving_log = pd.read_csv(simulator_output_folder + '/driving_log.csv', header=None)


def read_image(file):
    """ Return the image data from file path as numpy array"""
    image = Image.open(file)
    return np.asarray(image)


# Changing image file paths to simulator_output_folder
# Useful when the model is trained on a remote machine
driving_log[0] = driving_log[0].apply(lambda x: simulator_output_folder + '/IMG/' + x.split('/')[-1])
driving_log[1] = driving_log[1].apply(lambda x: simulator_output_folder + '/IMG/' + x.split('/')[-1])
driving_log[2] = driving_log[2].apply(lambda x: simulator_output_folder + '/IMG/' + x.split('/')[-1])

#Splitting data into training and validation set
from sklearn.cross_validation import train_test_split
data_train, data_test = train_test_split(driving_log, test_size=0.20, random_state=42)


# Defining the model for the task

model = Sequential()

# Crop top and bottom sections of the image
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Normalize the image
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# First convolution layer
model.add(Convolution2D(24, 5, 5,
                        border_mode='valid',
                        subsample=(2, 2)))
model.add(Activation('relu'))

# Second convolution layer
model.add(Convolution2D(36, 5, 5,
                        border_mode='valid',
                        subsample=(2, 2)))

model.add(Activation('relu'))

# Third convolution layer
model.add(Convolution2D(48, 5, 5,
                        border_mode='valid',
                        subsample=(2, 2)))
model.add(Activation('relu'))

# Fourth convolution layer
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid'))
model.add(Activation('relu'))


# Fifth convolution layer
model.add(Convolution2D(64, 3, 3,
                        border_mode='valid'))
model.add(Activation('relu'))

# Flatten output of previous layer to 1D array
model.add(Flatten())

# Dropout for reducing overfitting
model.add(Dropout(0.25))

# First fully connected layer
model.add(Dense(1164, activation="relu"))

# Second fully connected layer
model.add(Dense(100))
model.add(Activation('relu'))

# Third fully connected layer
model.add(Dense(50))

model.add(Activation('relu'))

# Fourth fully connected layer
model.add(Dense(10))

model.add(Activation('relu'))

# Output layer
model.add(Dense(1))

def input_generator(driving_log):
    """Get training data in batches of 129"""
    while 1:
        driving_log = driving_log.sample(frac=1)
        count = 0
        images = []
        steerings = []
        for row in driving_log.itertuples():
            image = read_image(row[1])
            images.append(image)
            steerings.append(row[4])
            #Left camera image
            image_left = read_image(row[2])
            images.append(image_left)
            steerings.append(row[4] + 0.2)
            #Right camera image
            image_right = read_image(row[3])
            images.append(image_right)
            steerings.append(row[4] - 0.2)
            count += 3
            if (count == 129):
                yield((np.asarray(images), np.asarray(steerings)))
                count = 0
                images = []
                steerings = []
        yield((np.asarray(images), np.asarray(steerings)))

def validation_generator(driving_log):
    """Get validation data in batches of 128"""
    while 1:
        driving_log = driving_log.sample(frac=1)
        count = 0
        images = []
        steerings = []
        for row in driving_log.itertuples():
            #Center
            image = read_image(row[1])
            images.append(image)
            steerings.append(row[4])
            count += 1
            if (count == 128):
                yield((np.asarray(images), np.asarray(steerings)))
                count = 0
                images = []
                steerings = []
        yield((np.asarray(images), np.asarray(steerings)))

# Train model using Adam optimizer and MSE loss
model.compile(Adam(lr=0.0001), 'mse', ['mean_squared_error'])
history = model.fit_generator(input_generator(data_train), samples_per_epoch=len(data_train)*3, nb_epoch=5, validation_data=validation_generator(data_test), nb_val_samples=len(data_test))

# Saving trained model for simulator
model.save('model.h5')



