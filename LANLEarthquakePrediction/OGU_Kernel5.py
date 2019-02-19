import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import cv2

import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

from threading import Thread
from multiprocessing.pool import ThreadPool

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
ACOUSTICDATA_NPY_PATH = NP_DATA_PATH + "\\acoustic_data.npy"
TIME_TO_FAILURE_NPY_PATH = NP_DATA_PATH + "\\time_to_failure.npy"
TRAIN_NP_PATH = NP_DATA_PATH + "\\train2d.npy"
MODEL_PATH = f"{DATA_PATH}\\models"

if os.path.exists(TRAIN_NP_PATH):
    train_data = np.load(TRAIN_NP_PATH)
    print("Traing 2d Np Files Loaded")
else:
    if os.path.exists(ACOUSTICDATA_NPY_PATH) and os.path.exists(TIME_TO_FAILURE_NPY_PATH):
        AcousticData = np.load(ACOUSTICDATA_NPY_PATH)
        TimeToFailure = np.load(TIME_TO_FAILURE_NPY_PATH)    
        print("Acoustic and Time To Fail Np Files Loaded")
    else:
        print("Build Acustic and Time To Fail")
        chunks = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, chunksize= 10 ** 6)    
        for chunk in tqdm(chunks):        
            AcousticData = np.append(AcousticData, np.array(chunk['acoustic_data']))
            TimeToFailure = np.append(TimeToFailure, np.array(chunk['time_to_failure']))

        np.save(ACOUSTICDATA_NPY_PATH, AcousticData)
        np.save(TIME_TO_FAILURE_NPY_PATH, TimeToFailure)
        print("Acoustic and Time To Fail Np Files Saved")

    print("Build 2d Train Data")
    train_data = np.column_stack((AcousticData, TimeToFailure))
    np.save(TRAIN_NP_PATH, train_data)
    print("Train 2d Np Files Saved")

#Build 150_000 row count samples
row_count = 150_000//5
row_dvider = 375//5//2

IMG_H = row_count//row_dvider;
IMG_W = row_dvider;
IMG_SIZE = 50

print("Creating Features")
features = np.reshape(train_data[:,0][len(train_data) % row_count:], (-1, IMG_H, IMG_W))
features = np.interp(features, (features.min(), features.max()), (0, +255)) #Scale between 0-255
samples = np.array([])
for sample in tqdm(features):    
    new_array = cv2.resize(sample, (IMG_SIZE, IMG_SIZE))
    samples = np.append(samples, (new_array))

print("Creating Labels")
labels = np.array([])
for i in tqdm(range(0,len(train_data),row_count)):
    part = train_data[i:][:row_count]    
    time_part = part[:,1]
    avg_time = np.average(time_part)    
    labels = np.append(labels, avg_time)

labels = labels[(len(labels)-len(features)):]

print("Label and Feture Shapes", labels.shape, features.shape)
#print(labels, features)

X = np.reshape(samples, (-1, IMG_SIZE, IMG_SIZE, 1))
X = X / 255.0
y = labels
print("Train X, Train y", X.shape, y.shape)

#BUILD CONV KERAS MODEL
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

BATCH_SIZE = 100
EPOCHS = 50

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:            
            modelName = f"epoch{EPOCHS}-resize{row_count}_{IMG_SIZE}-conv{conv_layer}-nodes{layer_size}-dense{dense_layer}"

            print(modelName)

            cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                            histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))            

            model.compile(loss='mae', optimizer='adam', metrics=['mae'])

            model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=[cb])
            model.save(f'{MODEL_PATH}\\{modelName}')