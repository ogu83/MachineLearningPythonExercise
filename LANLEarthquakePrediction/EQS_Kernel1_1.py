import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm as tqdm

from sklearn.linear_model import LinearRegression

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from scipy import integrate

import tables as tb

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib import style
style.use('ggplot')

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_SEG_PATH = f"{DATA_PATH}\\test"
TRAIN_SEG_PATH = f"{DATA_PATH}\\train"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"

TOTAL_ROW_COUNT = 629145480
TRAINING_DERIVED_ROW_COUNT = 150_000
segments = TOTAL_ROW_COUNT//TRAINING_DERIVED_ROW_COUNT

df_train_sum = pd.read_pickle(f"{PICKLE_PATH}\\df_train_sum.pickle")

print(df_train_sum.head())
print(df_train_sum.describe())
print(df_train_sum.time)