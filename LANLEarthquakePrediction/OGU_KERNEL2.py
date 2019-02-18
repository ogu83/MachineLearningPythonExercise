import numpy as np
import pandas as pd
import dask.dataframe as dd
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

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TRAIN_DATA_PATH_PRO = f"{DATA_PATH}\\pickle\\train_processed.pickle"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
ACOUSTICDATA_NPY_PATH = NP_DATA_PATH + "\\acoustic_data.npy"
TIME_TO_FAILURE_NPY_PATH = NP_DATA_PATH + "\\time_to_failure.npy"
TRAIN_NP_PATH = NP_DATA_PATH + "\\train2d.npy"
MODEL_PATH = f"{DATA_PATH}\\models"

#train = dd.read_csv(TRAIN_DATA_PATH)
train = pd.DataFrame()
chunks = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64}, chunksize= 10 ** 7, nrows=100_000_000)
for chunk in tqdm(chunks):    
   train = train.append(chunk)
print("Train Data Loaded")

train = train.to_sparse()
print("Processed Train Data To Sparse")
train = train.rename(columns={"acoustic_data": "signal", "time_to_failure": "quaketime"})
print("Processed Train Data Renamed")
stepsize = np.diff(train.quaketime)
print("Processed Train Data StepSize calculated")
train = train.drop(train.index[len(train)-1])
print("Processed Train Data Drop Action where no step size")
train["stepsize"] = stepsize
print("Processed Train Data StepSize added")
train.to_pickle(TRAIN_DATA_PATH_PRO)
print("Processed Train Data Saved")

cv = TimeSeriesSplit(n_splits=5)

#window_sizes = [10, 50, 100, 1000]
#for window in tqdm(window_sizes):
#    train["rolling_mean_" + str(window)] = train.signal.rolling(window=window).mean()
#    train["rolling_std_" + str(window)] = train.signal.rolling(window=window).std()
#    train["rolling_q25"] = train.signal.rolling(window=window).quantile(0.25)
#    train["rolling_q75"] = train.signal.rolling(window=window).quantile(0.75)
#    train["rolling_q50"] = train.signal.rolling(window=window).quantile(0.5)
#    train["rolling_iqr"] = train.rolling_q75 - train.rolling_q25
#    train["rolling_min"] = train.signal.rolling(window=window).min()
#    train["rolling_max"] = train.signal.rolling(window=window).max()
#    train["rolling_skew"] = train.signal.rolling(window=window).skew()
#    train["rolling_kurt"] = train.signal.rolling(window=window).kurt()

#train = train.dropna()

#print("Processed Train Data Saved")
#train.to_pickle(TRAIN_DATA_PATH_PRO)

print(train)
#print(train.describe())