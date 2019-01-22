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

