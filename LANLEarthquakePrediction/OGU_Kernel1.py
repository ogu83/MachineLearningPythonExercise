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
from keras.layers import Dense

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
    train_data = np.load(TRAIN_NP_PATH);
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

row_count = 150_000

print("Creating Fetures")
features = np.reshape(train_data[:,0][len(train_data) % row_count:], (-1, row_count))

print("Creating Labels")
labels = np.array([])
for i in tqdm(range(0,len(train_data),row_count)):
    part = train_data[i:][:row_count]    
    time_part = part[:,1]
    avg_time = np.average(time_part)    
    labels = np.append(labels, avg_time)

labels = labels[(len(labels)-len(features)):]

print("Label and Feture Shapes", labels.shape, features.shape)

##TRAIN MODEL WITH KERAS

EPOCHS = 100
BATCH_SIZE = 50
modelName="OGU_KERNEL_2048R_2014R_512R_256R_1"
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
    model.add(Dense(2048, input_dim=row_count, activation='relu'))    
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))      
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'accuracy'])
    model.fit(features, labels, validation_split = 0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)
    model.save(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Saved.", modelName)
print(model.summary())

##LOAD TEST DATA

##PREDICT WITH MODEL

##SAVE SUBMISSION

