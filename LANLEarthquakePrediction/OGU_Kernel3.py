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

def createFeatureCol(features_arr, colname):
    if colname == "avg":
        result = map(lambda x: np.average(x), features_arr)
    elif colname == "mean":
        result = map(lambda x: np.mean(x), features_arr)
    elif colname == "max":
        result = map(lambda x: np.max(x), features_arr)
    elif colname == "min":
        result = map(lambda x: np.min(x), features_arr)
    elif colname == "median":
        result = map(lambda x: np.median(x), features_arr)
    elif colname == "q25":
        result = map(lambda x: np.quantile(x,0.25), features_arr)
    elif colname == "q50":
        result = map(lambda x: np.quantile(x,0.50), features_arr)
    elif colname == "q75":
        result = map(lambda x: np.quantile(x,0.75), features_arr)

    result = np.fromiter(result, dtype=np.float64)    
    return result

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

window_size = 50
LABELS_PATH = f"{NP_DATA_PATH}\\ogu_kernel3_w{window_size}_labels.npy"
FEATURES_PATH = f"{NP_DATA_PATH}\\ogu_kernel3_w{window_size}_features.npy"
pool = ThreadPool(processes = 1)

if (os.path.exists(LABELS_PATH)):
    labels = np.load(LABELS_PATH)
else:
    window_labels_arr = np.reshape(train_data[:,1][len(train_data) % window_size:], (-1, window_size))
    labels = map(lambda x: np.average(x), window_labels_arr)
    labels = np.fromiter(labels, dtype=np.float64)
    np.save(LABELS_PATH,labels)

print("labels.shape", labels.shape)

if (os.path.exists(FEATURES_PATH)):
    features = np.load(FEATURES_PATH)
else:
    window_features_arr =  np.reshape(train_data[:,0][len(train_data) % window_size:], (-1, window_size))
    
    print("averages")
    averages = np.array([],dtype=np.float64)    
    thread_avg = pool.apply_async(createFeatureCol,(window_features_arr, "avg"))
    
    print("means")
    means = np.array([],dtype=np.float64)    
    thread_mean = pool.apply_async(createFeatureCol,(window_features_arr, "mean"))

    print("maxs")
    maxs = np.array([],dtype=np.float64)
    thread_max = pool.apply_async(createFeatureCol,(window_features_arr, "max"))

    print("mins")
    mins = np.array([],dtype=np.float64)
    thread_min = pool.apply_async(createFeatureCol,(window_features_arr, "min"))

    print("medians")
    medians = np.array([],dtype=np.float64)
    thread_medians = pool.apply_async(createFeatureCol,(window_features_arr, "median"))

    print("q25s")
    q25s = np.array([],dtype=np.float64)
    thread_q25s = pool.apply_async(createFeatureCol,(window_features_arr, "q25"))
    
    print("q75s")
    q75s = np.array([],dtype=np.float64)
    thread_q75s = pool.apply_async(createFeatureCol,(window_features_arr, "q75"))   

    averages = thread_avg.get()
    means = thread_mean.get()
    maxs = thread_max.get()
    mins = thread_min.get()
    medians = thread_medians.get()
    q25s = thread_q25s.get()
    q75s = thread_q75s.get()    

    print("Transposing Features")
    features = np.array([averages, means, maxs, mins, medians, q25s, q75s])
    features = np.transpose(features)

    np.save(FEATURES_PATH, features)

print("features.shape", features.shape);

##TRAIN MODEL WITH KERAS
EPOCHS = 100
BATCH_SIZE = 100
modelName="OGU_KERNEL_3_8R4R2R1L"
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
    model.compile(loss='mse', optimizer='adam', metrics=['mae','mse','accuracy'])
    model.fit(features, labels, validation_split = 0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)        
    model.save(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Saved.", modelName)
print(model.summary())

##LOAD TEST DATA
#submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id', dtype={"time_to_failure": np.float32})
#for i, seg_id in enumerate(tqdm(submission.index)):
#  #  print(i)
#    seg = pd.read_csv(TEST_DATA_PATH + "\\" + seg_id + '.csv')
#    x = np.array(seg['acoustic_data'].values)
#    x_features = np.reshape(x, (-1, row_count))
#    pred = model.predict(x_features)
#    print(x_features, pred)
#    submission.time_to_failure[i] = pred

#submission.head()

#submission.to_csv(f'{DATA_PATH}\\submission_{modelName}.csv')