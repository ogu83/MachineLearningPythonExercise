import tensorflow as tf
import os
import numpy as np
import time
import random
import keras
from tqdm import tqdm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

LOAD_TRAIN_FILES = False # true if we already have batch train files
LOAD_PREV_MODEL = False #true want to transfer learn from existing model
HALITE_TRESHOLD = 4400

TRAINING_CHUNK_SIZE = 500
PREV_MODEL_NAME = ""
VALIDATION_GAME_COUNT = 50
NAME = f"phase1-{int(time.time())}"
EPOCHS = 1

TRAINING_DATA_DIR = '/media/oguz/ParrotExt/HaliteTrainingData'
training_file_names = []

for f in os.listdir(TRAINING_DATA_DIR):
    halite_amount = int(f.split("-")[0])
    if halite_amount >= HALITE_TRESHOLD:
        training_file_names.append(os.path.join(TRAINING_DATA_DIR, f))

print(f"After the threshold we have  {len(training_file_names)} games.")

random.shuffle(training_file_names)

if LOAD_TRAIN_FILES:
    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")
else:
    test_x = []
    test_y = []

    for f in tqdm(training_file_names[:VALIDATION_GAME_COUNT]):
        data = np.load(f)

        for d in data:
            test_x.append(np.array(d[0]))
            test_y.append(d[1])

    np.save("text_x.npy")
    np.save("text_y.npy")

test_x = np.array(test_x)

#source: https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if LOAD_PREV_MODEL:
    model = tf.keras.models.load_model(PREV_MODEL_NAME)
else:
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=test_x.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(5))
    model.add(Activation('sigmoid'))

opt = tf.keras.optimizers.Adam(Lr=1e-3, decay=1e-3)
model.compile(Loss="sparse_categorical_crossentropy", optimizer=opt, metrics=['accuracy'])

