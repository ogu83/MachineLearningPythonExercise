import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"

train = pd.read_csv(TRAIN_DATA_PATH, nrows=1_000_000)
#print(train.head(5))

train.rename({"acoustic_data": "signal", "time_to_failure": "quaketime"}, axis="columns", inplace=True)
#print(train.head(5))
#print(train.describe())

stepsize = np.diff(train.quaketime)
train = train.drop(train.index[len(train)-1])
train["stepsize"] = stepsize
train.stepsize = train.stepsize.apply(lambda l: np.round(l, 10))
#print(train.head(5))

cv = TimeSeriesSplit(n_splits=5)

window_sizes = [10, 50, 100, 1000]
for window in window_sizes:
    train["rolling_mean_" + str(window)] = train.signal.rolling(window=window).mean()
    train["rolling_std_" + str(window)] = train.signal.rolling(window=window).std()
    train["rolling_q25"] = train.signal.rolling(window=window).quantile(0.25)
    train["rolling_q75"] = train.signal.rolling(window=window).quantile(0.75)
    train["rolling_q50"] = train.signal.rolling(window=window).quantile(0.5)
    train["rolling_iqr"] = train.rolling_q75 - train.rolling_q25
    train["rolling_min"] = train.signal.rolling(window=window).min()
    train["rolling_max"] = train.signal.rolling(window=window).max()
    train["rolling_skew"] = train.signal.rolling(window=window).skew()
    train["rolling_kurt"] = train.signal.rolling(window=window).kurt()

print(train)
