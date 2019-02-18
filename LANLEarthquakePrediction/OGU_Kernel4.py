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

import matplotlib.pyplot as plt

from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten

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

features = np.transpose(np.array([train_data[:,0]]))
labels = np.array(train_data[:,1])

print("labels.shape", "features.shape")
print(labels.shape, features.shape)

##TRAIN MODEL WITH KERAS
EPOCHS = 10
BATCH_SIZE = 1000
modelName="OGU_KERNEL_4_8R4R2R1L"
model = Sequential(name=modelName)
cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                            histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

if (os.path.exists(f"{MODEL_PATH}\\{model.name}.h5")):    
    model = load_model(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Loaded.", modelName)
else:    
    print("Training Model", modelName)    
    model.add(Dense(8, input_dim=features.shape[1], activation='relu'))    
    model.add(Dropout(0.1))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='relu'))                  
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    model.fit(features, labels, validation_split = 0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)        
    model.save(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Saved.", modelName)
print(model.summary())