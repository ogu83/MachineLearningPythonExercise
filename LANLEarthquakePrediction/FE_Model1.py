import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import lightgbm as lgb
import xgboost as xgb
import time
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
TRAINING_DERIVED_ROW_COUNT = 150_000
READ_WHOLE_TRAIN_DATA = True
READ_WHOLE_TEST_DATA = True
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"

Y_TRAIN_PICKLE = f"{PICKLE_PATH}\\y_train2.pickle"
X_TRAIN_PICKLE = f"{PICKLE_PATH}\\x_train2.pickle"
X_TEST_PICKLE = f"{PICKLE_PATH}\\x_test.pickle"

if os.path.exists(Y_TRAIN_PICKLE) and os.path.exists(X_TRAIN_PICKLE):
    X_train_scaled = pd.read_pickle(X_TRAIN_PICKLE)
    y_tr = pd.read_pickle(Y_TRAIN_PICKLE)
    print("Train Data Loaded")
    print(X_train_scaled.head())
    print(y_tr.head())
else:
    #READ TRAIN DATA
    if READ_WHOLE_TRAIN_DATA:
        train = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
    else:
        train = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows=500_000)

    rows = TRAINING_DERIVED_ROW_COUNT
    segments = int(np.floor(train.shape[0] / rows))
    segments1 = int(np.floor(train[rows//8:].shape[0] / rows))
    segments2 = int(np.floor(train[2*rows//8:].shape[0] / rows))
    segments3 = int(np.floor(train[3*rows//8:].shape[0] / rows))
    segments4 = int(np.floor(train[4*rows//8:].shape[0] / rows))
    segments5 = int(np.floor(train[5*rows//8:].shape[0] / rows))
    segments6 = int(np.floor(train[6*rows//8:].shape[0] / rows))
    segments7 = int(np.floor(train[7*rows//8:].shape[0] / rows))
    X_tr = pd.DataFrame(index=range(segments+segments1+segments2+segments3+segments4+segments5+segments6+segments7-1), dtype=np.float64)
    y_tr = pd.DataFrame(index=range(segments+segments1+segments2+segments3+segments4+segments5+segments6+segments7-1), dtype=np.float64, columns=['time_to_failure'])

    ittr = 0;    

    total_mean = train['acoustic_data'].mean()
    total_std = train['acoustic_data'].std()
    total_max = train['acoustic_data'].max()
    total_min = train['acoustic_data'].min()
    total_sum = train['acoustic_data'].sum()
    total_abs_sum = np.abs(train['acoustic_data']).sum()

    print("TOTALS")
    print(f"Mean: {total_mean}")
    print(f"Std: {total_std}")
    print(f"Max: {total_max}")
    print(f"Min: {total_min}")
    print(f"Sum: {total_sum}")
    print(f"Abs Sum: {total_abs_sum}")
    
    for tSegment in tqdm(range(0, 8)):
        train = train[rows//8:]
        segments = int(np.floor(train.shape[0] / rows))
        #Arrange X Training Data COLS and Values
        for segment in tqdm(range(segments)):
            seg = train.iloc[segment*rows:segment*rows+rows]
            x = pd.Series(seg['acoustic_data'].values)
            y = seg['time_to_failure'].values[-1]
    
            y_tr.loc[segment+ittr, 'time_to_failure'] = y
            X_tr.loc[segment+ittr, 'mean'] = x.mean()
            X_tr.loc[segment+ittr, 'std'] = x.std()
            X_tr.loc[segment+ittr, 'max'] = x.max()
            X_tr.loc[segment+ittr, 'min'] = x.min()
        
            X_tr.loc[segment+ittr, 'mean_change_abs'] = np.mean(np.diff(x))
            X_tr.loc[segment+ittr, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
            X_tr.loc[segment+ittr, 'abs_max'] = np.abs(x).max()
            X_tr.loc[segment+ittr, 'abs_min'] = np.abs(x).min()
    
            X_tr.loc[segment+ittr, 'std_first_50000'] = x[:50000].std()
            X_tr.loc[segment+ittr, 'std_last_50000'] = x[-50000:].std()
            X_tr.loc[segment+ittr, 'std_first_10000'] = x[:10000].std()
            X_tr.loc[segment+ittr, 'std_last_10000'] = x[-10000:].std()
    
            X_tr.loc[segment+ittr, 'avg_first_50000'] = x[:50000].mean()
            X_tr.loc[segment+ittr, 'avg_last_50000'] = x[-50000:].mean()
            X_tr.loc[segment+ittr, 'avg_first_10000'] = x[:10000].mean()
            X_tr.loc[segment+ittr, 'avg_last_10000'] = x[-10000:].mean()
    
            X_tr.loc[segment+ittr, 'min_first_50000'] = x[:50000].min()
            X_tr.loc[segment+ittr, 'min_last_50000'] = x[-50000:].min()
            X_tr.loc[segment+ittr, 'min_first_10000'] = x[:10000].min()
            X_tr.loc[segment+ittr, 'min_last_10000'] = x[-10000:].min()
    
            X_tr.loc[segment+ittr, 'max_first_50000'] = x[:50000].max()
            X_tr.loc[segment+ittr, 'max_last_50000'] = x[-50000:].max()
            X_tr.loc[segment+ittr, 'max_first_10000'] = x[:10000].max()
            X_tr.loc[segment+ittr, 'max_last_10000'] = x[-10000:].max()
    
            X_tr.loc[segment+ittr, 'max_to_min'] = x.max() / np.abs(x.min())
            X_tr.loc[segment+ittr, 'max_to_min_diff'] = x.max() - np.abs(x.min())
            X_tr.loc[segment+ittr, 'count_big'] = len(x[np.abs(x) > 500])
            X_tr.loc[segment+ittr, 'sum'] = x.sum()
    
            X_tr.loc[segment+ittr, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
            X_tr.loc[segment+ittr, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
            X_tr.loc[segment+ittr, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
            X_tr.loc[segment+ittr, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
    
            X_tr.loc[segment+ittr, 'q95'] = np.quantile(x, 0.95)
            X_tr.loc[segment+ittr, 'q99'] = np.quantile(x, 0.99)
            X_tr.loc[segment+ittr, 'q05'] = np.quantile(x, 0.05)
            X_tr.loc[segment+ittr, 'q01'] = np.quantile(x, 0.01)
    
            X_tr.loc[segment+ittr, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
            X_tr.loc[segment+ittr, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
            X_tr.loc[segment+ittr, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
            X_tr.loc[segment+ittr, 'abs_q01'] = np.quantile(np.abs(x), 0.01)
    
            X_tr.loc[segment+ittr, 'trend'] = add_trend_feature(x)
            X_tr.loc[segment+ittr, 'abs_trend'] = add_trend_feature(x, abs_values=True)
            X_tr.loc[segment+ittr, 'abs_mean'] = np.abs(x).mean()
            X_tr.loc[segment+ittr, 'abs_std'] = np.abs(x).std()
    
            X_tr.loc[segment+ittr, 'mad'] = x.mad()
            X_tr.loc[segment+ittr, 'kurt'] = x.kurtosis()
            X_tr.loc[segment+ittr, 'skew'] = x.skew()
            X_tr.loc[segment+ittr, 'med'] = x.median()
    
            X_tr.loc[segment+ittr, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
            X_tr.loc[segment+ittr, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
            X_tr.loc[segment+ittr, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
            X_tr.loc[segment+ittr, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
            X_tr.loc[segment+ittr, 'Moving_average_1500_mean'] = x.rolling(window=1500).mean().mean(skipna=True)
            X_tr.loc[segment+ittr, 'Moving_average_3000_mean'] = x.rolling(window=3000).mean().mean(skipna=True)
            X_tr.loc[segment+ittr, 'Moving_average_6000_mean'] = x.rolling(window=6000).mean().mean(skipna=True)
            ewma = pd.Series.ewm
            X_tr.loc[segment+ittr, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
            X_tr.loc[segment+ittr, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
            X_tr.loc[segment+ittr, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
            no_of_std = 3
            X_tr.loc[segment+ittr, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
            X_tr.loc[segment+ittr,'MA_700MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
            X_tr.loc[segment+ittr,'MA_700MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
            X_tr.loc[segment+ittr, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
            X_tr.loc[segment+ittr,'MA_400MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
            X_tr.loc[segment+ittr,'MA_400MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
            X_tr.loc[segment+ittr, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    
            X_tr.loc[segment+ittr, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
            X_tr.loc[segment+ittr, 'q999'] = np.quantile(x,0.999)
            X_tr.loc[segment+ittr, 'q001'] = np.quantile(x,0.001)
            X_tr.loc[segment+ittr, 'ave10'] = stats.trim_mean(x, 0.1)
    
            for windows in [10, 100, 1000]:
                x_roll_std = x.rolling(windows).std().dropna().values
                x_roll_mean = x.rolling(windows).mean().dropna().values                
        
                X_tr.loc[segment+ittr, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
                X_tr.loc[segment+ittr, 'std_roll_std_' + str(windows)] = x_roll_std.std()
                X_tr.loc[segment+ittr, 'max_roll_std_' + str(windows)] = x_roll_std.max()
                X_tr.loc[segment+ittr, 'min_roll_std_' + str(windows)] = x_roll_std.min()
                X_tr.loc[segment+ittr, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
                X_tr.loc[segment+ittr, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)        
                X_tr.loc[segment+ittr, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
                X_tr.loc[segment+ittr, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
                X_tr.loc[segment+ittr, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
                X_tr.loc[segment+ittr, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
                X_tr.loc[segment+ittr, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
                X_tr.loc[segment+ittr, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
                X_tr.loc[segment+ittr, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
                X_tr.loc[segment+ittr, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
                X_tr.loc[segment+ittr, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
                X_tr.loc[segment+ittr, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
                X_tr.loc[segment+ittr, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)        
                X_tr.loc[segment+ittr, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
                X_tr.loc[segment+ittr, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
                X_tr.loc[segment+ittr, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
                X_tr.loc[segment+ittr, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
                X_tr.loc[segment+ittr, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
        ittr += segments
       
    print(f'{X_tr.shape[0]} samples in new train data and {X_tr.shape[1]} columns.')    

    np.abs(X_tr.corrwith(y_tr['time_to_failure'])).sort_values(ascending=False).head(12)

    # fillna in new columns
    classic_sta_lta5_mean_fill = X_tr.loc[X_tr['classic_sta_lta5_mean'] != -np.inf, 'classic_sta_lta5_mean'].mean()
    X_tr.loc[X_tr['classic_sta_lta5_mean'] == -np.inf, 'classic_sta_lta5_mean'] = classic_sta_lta5_mean_fill
    X_tr['classic_sta_lta5_mean'] = X_tr['classic_sta_lta5_mean'].fillna(classic_sta_lta5_mean_fill)
    classic_sta_lta7_mean_fill = X_tr.loc[X_tr['classic_sta_lta7_mean'] != -np.inf, 'classic_sta_lta7_mean'].mean()
    X_tr.loc[X_tr['classic_sta_lta7_mean'] == -np.inf, 'classic_sta_lta7_mean'] = classic_sta_lta7_mean_fill
    X_tr['classic_sta_lta7_mean'] = X_tr['classic_sta_lta7_mean'].fillna(classic_sta_lta7_mean_fill)

    scaler = StandardScaler()
    scaler.fit(X_tr)
    X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
    X_train_scaled.to_pickle(X_TRAIN_PICKLE)    
    print("x_train_scaled.pickle saved")
    y_tr.to_pickle(Y_TRAIN_PICKLE)
    print("y_train_scaled.pickle saved")

if os.path.exists(X_TEST_PICKLE):
    X_test_scaled = pd.read_pickle(X_TEST_PICKLE)
    print("X Test Data Loaded")
    print(X_test_scaled.head())
else:
    #READING TEST DATA
    if (READ_WHOLE_TEST_DATA):
        submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
    else:
        submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id', nrows = 3)

    X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)
    plt.figure(figsize=(22, 16))

    #READ ALL SEGMENTS ACUSTIC DATA AND ARRANGE COLS AND VALUES
    for i, seg_id in enumerate(tqdm(X_test.index)):
        seg = pd.read_csv(TEST_DATA_PATH + '\\' + seg_id + '.csv')
    
        x = pd.Series(seg['acoustic_data'].values)
        X_test.loc[seg_id, 'mean'] = x.mean()
        X_test.loc[seg_id, 'std'] = x.std()
        X_test.loc[seg_id, 'max'] = x.max()
        X_test.loc[seg_id, 'min'] = x.min()
        
        X_test.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))
        X_test.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
        X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
        X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()
    
        X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
        X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
        X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
        X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()
    
        X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()
        X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()
        X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()
        X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()
    
        X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()
        X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()
        X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()
        X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()
    
        X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()
        X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()
        X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()
        X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()
    
        X_test.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())
        X_test.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())
        X_test.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])
        X_test.loc[seg_id, 'sum'] = x.sum()
    
        X_test.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
        X_test.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
        X_test.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
        X_test.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
    
        X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
        X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
        X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)
        X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)
    
        X_test.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(x), 0.95)
        X_test.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(x), 0.99)
        X_test.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(x), 0.05)
        X_test.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(x), 0.01)
    
        X_test.loc[seg_id, 'trend'] = add_trend_feature(x)
        X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)
        X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
        X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()
    
        X_test.loc[seg_id, 'mad'] = x.mad()
        X_test.loc[seg_id, 'kurt'] = x.kurtosis()
        X_test.loc[seg_id, 'skew'] = x.skew()
        X_test.loc[seg_id, 'med'] = x.median()
    
        X_test.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
        X_test.loc[seg_id, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
        X_test.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        X_test.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        X_test.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        X_test.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        X_test.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        X_test.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        X_test.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        X_test.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()
        X_test.loc[seg_id, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
        X_test.loc[seg_id, 'Moving_average_1500_mean'] = x.rolling(window=1500).mean().mean(skipna=True)
        X_test.loc[seg_id, 'Moving_average_3000_mean'] = x.rolling(window=3000).mean().mean(skipna=True)
        X_test.loc[seg_id, 'Moving_average_6000_mean'] = x.rolling(window=6000).mean().mean(skipna=True)
        ewma = pd.Series.ewm
        X_test.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
        X_test.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
        X_test.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
        no_of_std = 3
        X_test.loc[seg_id, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
        X_test.loc[seg_id,'MA_700MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
        X_test.loc[seg_id,'MA_700MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
        X_test.loc[seg_id, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
        X_test.loc[seg_id,'MA_400MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
        X_test.loc[seg_id,'MA_400MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
        X_test.loc[seg_id, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    
        X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)
        X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)
        X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)
    
        for windows in [10, 100, 1000]:
            x_roll_std = x.rolling(windows).std().dropna().values
            x_roll_mean = x.rolling(windows).mean().dropna().values
        
            X_test.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
            X_test.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
            X_test.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
            X_test.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
            X_test.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
            X_test.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
            X_test.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
            X_test.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
            X_test.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
            X_test.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            X_test.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
            X_test.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
            X_test.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
            X_test.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
            X_test.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
            X_test.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            X_test.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            X_test.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            X_test.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            X_test.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
            X_test.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            X_test.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    
        if i < 12:
            plt.subplot(6, 4, i + 1)
            plt.plot(seg['acoustic_data'])
            plt.title(seg_id)
    
    # fillna in new columns
    X_test.loc[X_test['classic_sta_lta5_mean'] == -np.inf, 'classic_sta_lta5_mean'] = classic_sta_lta5_mean_fill
    X_test.loc[X_test['classic_sta_lta7_mean'] == -np.inf, 'classic_sta_lta7_mean'] = classic_sta_lta7_mean_fill
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_test_scaled.to_pickle(X_TEST_PICKLE)
    print("x_test_scaled.pickle saved")
    
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        if model_type == 'rndForest':
            model = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                                            max_features=0.5, #Max number of features each tree can use 
                                            min_samples_leaf=30, #Min amount of samples in each leaf
                                            random_state=11)
            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'RandomForestRegressor: Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=True)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction

# Taking less columns seriously decreases score.
# top_cols = list(feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index)
# X_train_scaled = X_train_scaled[top_cols]
# X_test_scaled = X_test_scaled[top_cols]

params = {
    'num_leaves': 54,
    'min_data_in_leaf': 79,
    'objective': 'huber',
    'max_depth': -1,
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "bagging_freq": 5,
    "bagging_fraction": 0.8126672064208567,
    "bagging_seed": 11,
    "metric": 'mae',
    "verbosity": -1,
    'reg_alpha': 0.1302650970728192,
    'reg_lambda': 0.3603427518866501
}
print("******LGB Started******")
oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)

xgb_params = {'eta': 0.05,
              'max_depth': 10,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4}
print("******XGB Started******")
oof_xgb, prediction_xgb = train_model(X=X_train_scaled, X_test=X_test_scaled, params=xgb_params, model_type='xgb')

params = {'loss_function':'MAE'}
print("CAT Started")
oof_cat, prediction_cat = train_model(X=X_train_scaled, X_test=X_test_scaled, params=params, model_type='cat')

model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
print("******Sklearn Started******")
oof_r, prediction_r = train_model(X=X_train_scaled, X_test=X_test_scaled, params=None, model_type='sklearn', model=model)

#train_stack = np.vstack([oof_lgb, oof_xgb, oof_r, oof_cat]).transpose()
#train_stack = pd.DataFrame(train_stack, columns = ['lgb', 'xgb', 'r', 'cat'])
#test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_r, prediction_cat]).transpose()
#test_stack = pd.DataFrame(test_stack)
#print("******LGB_STACK STARTED******")
#oof_lgb_stack, prediction_lgb_stack, feature_importance = train_model(X=train_stack, X_test=test_stack, params=params, model_type='lgb', plot_feature_importance=True)

#MAKE SUBMISSION FILE 
if (READ_WHOLE_TEST_DATA):
    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
else:
    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id', nrows = 3)

submission['time_to_failure'] = (prediction_lgb + prediction_xgb + prediction_r + prediction_cat) / 4
# submission['time_to_failure'] = prediction_lgb_stack
print(submission.head())
submission.to_csv(f'{DATA_PATH}\\submission.csv')