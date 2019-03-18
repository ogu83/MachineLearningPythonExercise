import pandas as pd
pd.options.mode.use_inf_as_na = True
import numpy as np
import os
import pickle
from tqdm import tqdm as tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from scipy import integrate

#import tables as tb

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from matplotlib import style
style.use('ggplot')

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization,Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
import keras

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_SEG_PATH = f"{DATA_PATH}\\test"
TRAIN_SEG_PATH = f"{DATA_PATH}\\train"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"
MODEL_PATH = f"{DATA_PATH}\\models"

TOTAL_ROW_COUNT = 629145480
TRAINING_DERIVED_ROW_COUNT = 150_000
segments = TOTAL_ROW_COUNT//TRAINING_DERIVED_ROW_COUNT

'''Numerically integrate the time series.
    @param how: the method to use (trapz by default)
    @return 
    Available methods:
     * trapz - trapezoidal
     * cumtrapz - cumulative trapezoidal
     * simps - Simpson's rule
     * romb - Romberger's rule     
'''
def integrate_cumtrapz(x, how='cumtrapz',i=0):    
    y = x
    return integrate.cumtrapz(y, x, initial=i)

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

windows = [10, 30, 50, 100, 150, 300, 500, 1000, 1500, 3000, 10000, 15000, 30000,50000]
def get_column_names():
    keys = list(['disp','vel','acc'])
    functions = list(['mean','median','max','std','var','sum','skew','kurt'])
    window_keys = ['mean','std']
    additional = []
    columns=[]
    for k in keys:
        for f in functions:
            columns.append(f"{k}_{f}")
            for wk in window_keys:
                for w in windows:                
                    columns.append(f"{k}_{wk}_{f}_{w}")
            
    return columns

if not os.path.exists(f"{PICKLE_PATH}\\df_train_sum.pickle"):
    df_train = pd.DataFrame(columns=['time','acc','vel','disp'])
    for i in tqdm(range(5)):
        df_chunk = pd.read_pickle(f"{PICKLE_PATH}\\Samples{i*1000}.pickle")
        df_train = df_train.append(df_chunk, ignore_index=True)    

    print(df_train.head())
    #for i in range(0,2):
    #    fig = plt.figure(constrained_layout=True)
    #    gs = gridspec.GridSpec(1, 2, figure=fig)

    #    ax = fig.add_subplot(gs[0, 0])
    #    ax.plot(df_train[i:i+1]["disp"].values[0])
    #    ax.set_ylabel('disp')

    #    ax1 = fig.add_subplot(gs[0, 1])
    #    ax1.plot(df_train[i:i+1]["vel"].values[0])
    #    ax1.set_ylabel('velocity')

    #    #ax2 = fig.add_subplot(gs[0, 2])
    #    #ax2.plot(df_train[i:i+1]["acc"].values[0])
    #    #ax2.set_ylabel('acc')

    #    plt.title(f'{df_train[i:i+1]["time"].values[0]} sn')
    #    plt.show()
       
    df_train_sum = pd.DataFrame(index=range(segments), columns=get_column_names())
    df_train_sum["time"] = df_train["time"]

    print((df_train_sum.columns))

    for segment in tqdm(range(segments)):    
        disp = df_train.loc[segment, 'disp']    
        disp = pd.Series(disp)
        df_train_sum.loc[segment,'disp_mean']=disp.mean()
        df_train_sum.loc[segment,'disp_median']=disp.median()
        df_train_sum.loc[segment,'disp_max']=disp.max()
        df_train_sum.loc[segment,'disp_min']=disp.min()
        df_train_sum.loc[segment,'disp_std']=disp.std()
        df_train_sum.loc[segment,'disp_var']=disp.var()
        df_train_sum.loc[segment,'disp_sum']=disp.sum()    
        df_train_sum.loc[segment,'disp_skew']=disp.skew()
        df_train_sum.loc[segment,'disp_kurt']=disp.kurt()
    
        vel = df_train.loc[segment, 'vel']
        vel = pd.Series(vel)
        df_train_sum.loc[segment,'vel_mean']=vel.mean()
        df_train_sum.loc[segment,'vel_median']=vel.median()
        df_train_sum.loc[segment,'vel_max']=vel.max()
        df_train_sum.loc[segment,'vel_min']=vel.min()
        df_train_sum.loc[segment,'vel_std']=vel.std()
        df_train_sum.loc[segment,'vel_var']=vel.var()
        df_train_sum.loc[segment,'vel_sum']=vel.sum()    
        df_train_sum.loc[segment,'vel_skew']=vel.skew()
        df_train_sum.loc[segment,'vel_kurt']=vel.kurt()

        acc = df_train.loc[segment, 'acc']
        acc = pd.Series(vel)
        df_train_sum.loc[segment,'acc_mean']=acc.mean()
        df_train_sum.loc[segment,'acc_median']=acc.median()
        df_train_sum.loc[segment,'acc_max']=acc.max()
        df_train_sum.loc[segment,'acc_min']=acc.min()
        df_train_sum.loc[segment,'acc_std']=acc.std()
        df_train_sum.loc[segment,'acc_var']=acc.var()
        df_train_sum.loc[segment,'acc_sum']=acc.sum()    
        df_train_sum.loc[segment,'acc_skew']=acc.skew()
        df_train_sum.loc[segment,'acc_kurt']=acc.kurt()


        df_train_sum.loc[segment, 'mean_change_abs_acc'] = np.mean(np.diff(acc))
        df_train_sum.loc[segment, 'mean_change_abs_vel'] = np.mean(np.diff(vel))
        df_train_sum.loc[segment, 'mean_change_abs_disp'] = np.mean(np.diff(disp))

        df_train_sum.loc[segment, 'mean_change_rate_acc'] = np.mean(np.nonzero((np.diff(acc) / acc[:-1]))[0])
        df_train_sum.loc[segment, 'mean_change_rate_vel'] = np.mean(np.nonzero((np.diff(vel) / vel[:-1]))[0])
        df_train_sum.loc[segment, 'mean_change_rate_disp'] = np.mean(np.nonzero((np.diff(disp) / disp[:-1]))[0])
                
        df_train_sum.loc[segment, 'abs_max_acc'] = np.abs(acc).max()
        df_train_sum.loc[segment, 'abs_max_vel'] = np.abs(vel).max()
        df_train_sum.loc[segment, 'abs_max_disp'] = np.abs(disp).max()

        df_train_sum.loc[segment, 'abs_min_acc'] = np.abs(acc).min()
        df_train_sum.loc[segment, 'abs_min_vel'] = np.abs(vel).min()
        df_train_sum.loc[segment, 'abs_min_disp'] = np.abs(disp).min()

        df_train_sum.loc[segment, 'max_to_min_acc'] = acc.max() / np.abs(acc.min())
        df_train_sum.loc[segment, 'max_to_min_vel'] = vel.max() / np.abs(vel.min())
        df_train_sum.loc[segment, 'max_to_min_disp'] = disp.max() / np.abs(disp.min())

        df_train_sum.loc[segment, 'max_to_min_diff_acc'] = acc.max() - np.abs(acc.min())
        df_train_sum.loc[segment, 'max_to_min_diff_vel'] = vel.max() - np.abs(vel.min())
        df_train_sum.loc[segment, 'max_to_min_diff_disp'] = disp.max() - np.abs(disp.min())

        df_train_sum.loc[segment, 'q95_acc'] = np.quantile(acc,0.95)
        df_train_sum.loc[segment, 'q99_acc'] = np.quantile(acc,0.99)
        df_train_sum.loc[segment, 'q05_acc'] = np.quantile(acc,0.05)
        df_train_sum.loc[segment, 'q01_acc'] = np.quantile(acc,0.01)

        df_train_sum.loc[segment, 'q95_vel'] = np.quantile(vel,0.95)
        df_train_sum.loc[segment, 'q99_vel'] = np.quantile(vel,0.99)
        df_train_sum.loc[segment, 'q05_vel'] = np.quantile(vel,0.05)
        df_train_sum.loc[segment, 'q01_vel'] = np.quantile(vel,0.01)

        df_train_sum.loc[segment, 'q95_disp'] = np.quantile(disp,0.95)
        df_train_sum.loc[segment, 'q99_disp'] = np.quantile(disp,0.99)
        df_train_sum.loc[segment, 'q05_disp'] = np.quantile(disp,0.05)
        df_train_sum.loc[segment, 'q01_disp'] = np.quantile(disp,0.01)

        df_train_sum.loc[segment, 'abs_q95_acc'] = np.quantile(np.abs(acc), 0.95)
        df_train_sum.loc[segment, 'abs_q99_acc'] = np.quantile(np.abs(acc), 0.99)
        df_train_sum.loc[segment, 'abs_q05_acc'] = np.quantile(np.abs(acc), 0.05)
        df_train_sum.loc[segment, 'abs_q01_acc'] = np.quantile(np.abs(acc), 0.01)

        df_train_sum.loc[segment, 'abs_q95_vel'] = np.quantile(np.abs(vel), 0.95)
        df_train_sum.loc[segment, 'abs_q99_vel'] = np.quantile(np.abs(vel), 0.99)
        df_train_sum.loc[segment, 'abs_q05_vel'] = np.quantile(np.abs(vel), 0.05)
        df_train_sum.loc[segment, 'abs_q01_vel'] = np.quantile(np.abs(vel), 0.01)

        df_train_sum.loc[segment, 'abs_q95_disp'] = np.quantile(np.abs(disp), 0.95)
        df_train_sum.loc[segment, 'abs_q99_disp'] = np.quantile(np.abs(disp), 0.99)
        df_train_sum.loc[segment, 'abs_q05_disp'] = np.quantile(np.abs(disp), 0.05)
        df_train_sum.loc[segment, 'abs_q01_disp'] = np.quantile(np.abs(disp), 0.01)

        df_train_sum.loc[segment, 'trend_acc'] = add_trend_feature(acc)
        df_train_sum.loc[segment, 'abs_trend_acc'] = add_trend_feature(acc, abs_values=True)
        df_train_sum.loc[segment, 'abs_mean_acc'] = np.abs(acc).mean()
        df_train_sum.loc[segment, 'abs_std_acc'] = np.abs(acc).std()

        df_train_sum.loc[segment, 'trend_vel'] = add_trend_feature(vel)
        df_train_sum.loc[segment, 'abs_trend_vel'] = add_trend_feature(vel, abs_values=True)
        df_train_sum.loc[segment, 'abs_mean_vel'] = np.abs(vel).mean()
        df_train_sum.loc[segment, 'abs_std_vel'] = np.abs(vel).std()

        df_train_sum.loc[segment, 'trend_disp'] = add_trend_feature(disp)
        df_train_sum.loc[segment, 'abs_trend_disp'] = add_trend_feature(disp, abs_values=True)
        df_train_sum.loc[segment, 'abs_mean_disp'] = np.abs(disp).mean()
        df_train_sum.loc[segment, 'abs_std_disp'] = np.abs(disp).std()

        df_train_sum.loc[segment, 'Hilbert_mean_acc'] = np.abs(hilbert(acc)).mean()
        df_train_sum.loc[segment, 'Hann_window_mean_acc'] = (convolve(acc, hann(150), mode='same') / sum(hann(150))).mean()

        df_train_sum.loc[segment, 'Hilbert_mean_vel'] = np.abs(hilbert(vel)).mean()
        df_train_sum.loc[segment, 'Hann_window_mean_vel'] = (convolve(vel, hann(150), mode='same') / sum(hann(150))).mean()

        df_train_sum.loc[segment, 'Hilbert_mean_disp'] = np.abs(hilbert(disp)).mean()
        df_train_sum.loc[segment, 'Hann_window_mean_disp'] = (convolve(disp, hann(150), mode='same') / sum(hann(150))).mean()               
        
        for w in tqdm(windows):                
            df_train_sum.loc[segment,f'disp_mean_mean_{w}'] = disp.rolling(window=w).mean().mean()
            df_train_sum.loc[segment,f'disp_mean_median_{w}'] = disp.rolling(window=w).mean().median()
            df_train_sum.loc[segment,f'disp_mean_max_{w}'] = disp.rolling(window=w).mean().max()
            df_train_sum.loc[segment,f'disp_mean_min_{w}'] = disp.rolling(window=w).mean().min()
            df_train_sum.loc[segment,f'disp_mean_std_{w}'] = disp.rolling(window=w).mean().std()
            df_train_sum.loc[segment,f'disp_mean_var_{w}'] = disp.rolling(window=w).mean().var()
            df_train_sum.loc[segment,f'disp_mean_sum_{w}'] = disp.rolling(window=w).mean().sum()
            df_train_sum.loc[segment,f'disp_mean_skew_{w}'] = disp.rolling(window=w).mean().skew()
            df_train_sum.loc[segment,f'disp_mean_kurt_{w}'] = disp.rolling(window=w).mean().kurt()

            df_train_sum.loc[segment,f'disp_std_mean_{w}'] = disp.rolling(window=w).std().mean()
            df_train_sum.loc[segment,f'disp_std_median_{w}'] = disp.rolling(window=w).std().median()
            df_train_sum.loc[segment,f'disp_std_max_{w}'] = disp.rolling(window=w).std().max()
            df_train_sum.loc[segment,f'disp_std_min_{w}'] = disp.rolling(window=w).std().min()
            df_train_sum.loc[segment,f'disp_std_std_{w}'] = disp.rolling(window=w).std().std()
            df_train_sum.loc[segment,f'disp_std_var_{w}'] = disp.rolling(window=w).std().var()
            df_train_sum.loc[segment,f'disp_std_sum_{w}'] = disp.rolling(window=w).std().sum()
            df_train_sum.loc[segment,f'disp_std_skew_{w}'] = disp.rolling(window=w).std().skew()
            df_train_sum.loc[segment,f'disp_std_kurt_{w}'] = disp.rolling(window=w).std().kurt()


            df_train_sum.loc[segment,f'vel_mean_mean_{w}'] = vel.rolling(window=w).mean().mean()
            df_train_sum.loc[segment,f'vel_mean_median_{w}'] = vel.rolling(window=w).mean().median()
            df_train_sum.loc[segment,f'vel_mean_max_{w}'] = vel.rolling(window=w).mean().max()
            df_train_sum.loc[segment,f'vel_mean_min_{w}'] = vel.rolling(window=w).mean().min()
            df_train_sum.loc[segment,f'vel_mean_std_{w}'] = vel.rolling(window=w).mean().std()
            df_train_sum.loc[segment,f'vel_mean_var_{w}'] = vel.rolling(window=w).mean().var()
            df_train_sum.loc[segment,f'vel_mean_sum_{w}'] = vel.rolling(window=w).mean().sum()
            df_train_sum.loc[segment,f'vel_mean_skew_{w}'] = vel.rolling(window=w).mean().skew()
            df_train_sum.loc[segment,f'vel_mean_kurt_{w}'] = vel.rolling(window=w).mean().kurt() 
        
            df_train_sum.loc[segment,f'vel_std_mean_{w}'] = vel.rolling(window=w).std().mean()
            df_train_sum.loc[segment,f'vel_std_median_{w}'] = vel.rolling(window=w).std().median()
            df_train_sum.loc[segment,f'vel_std_max_{w}'] = vel.rolling(window=w).std().max()
            df_train_sum.loc[segment,f'vel_std_min_{w}'] = vel.rolling(window=w).std().min()
            df_train_sum.loc[segment,f'vel_std_std_{w}'] = vel.rolling(window=w).std().std()
            df_train_sum.loc[segment,f'vel_std_var_{w}'] = vel.rolling(window=w).std().var()
            df_train_sum.loc[segment,f'vel_std_sum_{w}'] = vel.rolling(window=w).std().sum()
            df_train_sum.loc[segment,f'vel_std_skew_{w}'] = vel.rolling(window=w).std().skew()
            df_train_sum.loc[segment,f'vel_std_kurt_{w}'] = vel.rolling(window=w).std().kurt()                


            df_train_sum.loc[segment,f'acc_mean_mean_{w}'] = acc.rolling(window=w).mean().mean()
            df_train_sum.loc[segment,f'acc_mean_median_{w}'] = acc.rolling(window=w).mean().median()
            df_train_sum.loc[segment,f'acc_mean_max_{w}'] = acc.rolling(window=w).mean().max()
            df_train_sum.loc[segment,f'acc_mean_min_{w}'] = acc.rolling(window=w).mean().min()
            df_train_sum.loc[segment,f'acc_mean_std_{w}'] = acc.rolling(window=w).mean().std()
            df_train_sum.loc[segment,f'acc_mean_var_{w}'] = acc.rolling(window=w).mean().var()
            df_train_sum.loc[segment,f'acc_mean_sum_{w}'] = acc.rolling(window=w).mean().sum()
            df_train_sum.loc[segment,f'acc_mean_skew_{w}'] = acc.rolling(window=w).mean().skew()
            df_train_sum.loc[segment,f'acc_mean_kurt_{w}'] = acc.rolling(window=w).mean().kurt()   

            df_train_sum.loc[segment,f'acc_std_mean_{w}'] = acc.rolling(window=w).std().mean()
            df_train_sum.loc[segment,f'acc_std_median_{w}'] = acc.rolling(window=w).std().median()
            df_train_sum.loc[segment,f'acc_std_max_{w}'] = acc.rolling(window=w).std().max()
            df_train_sum.loc[segment,f'acc_std_min_{w}'] = acc.rolling(window=w).std().min()
            df_train_sum.loc[segment,f'acc_std_std_{w}'] = acc.rolling(window=w).std().std()
            df_train_sum.loc[segment,f'acc_std_var_{w}'] = acc.rolling(window=w).std().var()
            df_train_sum.loc[segment,f'acc_std_sum_{w}'] = acc.rolling(window=w).std().sum()
            df_train_sum.loc[segment,f'acc_std_skew_{w}'] = acc.rolling(window=w).std().skew()
            df_train_sum.loc[segment,f'acc_std_kurt_{w}'] = acc.rolling(window=w).std().kurt()                
            
    print(df_train_sum)
    df_train_sum.to_pickle(f"{PICKLE_PATH}\\df_train_sum.pickle")
    print("df_train_sum saved")
    del df_train
else:
    df_train_sum = pd.read_pickle(f"{PICKLE_PATH}\\df_train_sum.pickle")
    print("df_train_sum loaded")

submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
if not os.path.exists(f"{PICKLE_PATH}\\df_test_sum.pickle"):    
    df_test = pd.DataFrame(index=submission.index, columns=['time','acc','vel','disp'])
    df_test_sum = pd.DataFrame(columns=df_train_sum.columns, dtype=np.float64, index=submission.index)    
    
    last_vel = 0
    last_disp = 0    
    for i, seg_id in enumerate(tqdm(df_test.index)):
        seg = pd.read_csv(TEST_SEG_PATH + "\\" + seg_id + '.csv')           
    
        acc = seg['acoustic_data'].values
        df_test.loc[seg_id, 'acc'] = acc
    
        vel = integrate_cumtrapz(acc, last_vel)
        last_vel = vel[-1:]
        df_test.loc[seg_id, 'vel'] = vel
        
        disp = integrate_cumtrapz(vel, last_disp)    
        last_disp = disp[-1:]
        df_test.loc[seg_id, 'disp'] = disp

    #for i in range(0,2):
    #    fig = plt.figure(constrained_layout=True)
    #    gs = gridspec.GridSpec(1, 2, figure=fig)

    #    ax = fig.add_subplot(gs[0, 0])
    #    ax.plot(df_test[i:i+1]["disp"].values[0])
    #    ax.set_ylabel('disp')

    #    ax1 = fig.add_subplot(gs[0, 1])
    #    ax1.plot(df_test[i:i+1]["vel"].values[0])
    #    ax1.set_ylabel('velocity')

    #    #ax2 = fig.add_subplot(gs[0, 2])
    #    #ax2.plot(df_train[i:i+1]["acc"].values[0])
    #    #ax2.set_ylabel('acc')

    #    plt.title(f'{df_test[i:i+1]["time"].values[0]} sn')
    #    plt.show()

    #for segment in tqdm(range(segments)):   
    for i, seg_id in enumerate(tqdm(df_test.index)):       
        disp = df_test.loc[seg_id, 'disp']    
        disp = pd.Series(disp)
        df_test_sum.loc[seg_id,'disp_mean']=disp.mean()
        df_test_sum.loc[seg_id,'disp_median']=disp.median()
        df_test_sum.loc[seg_id,'disp_max']=disp.max()
        df_test_sum.loc[seg_id,'disp_min']=disp.min()
        df_test_sum.loc[seg_id,'disp_std']=disp.std()
        df_test_sum.loc[seg_id,'disp_var']=disp.var()
        df_test_sum.loc[seg_id,'disp_sum']=disp.sum()    
        df_test_sum.loc[seg_id,'disp_skew']=disp.skew()
        df_test_sum.loc[seg_id,'disp_kurt']=disp.kurt()
    
        vel = df_test.loc[seg_id, 'vel']    
        vel = pd.Series(vel)
        df_test_sum.loc[seg_id,'vel_mean']=vel.mean()
        df_test_sum.loc[seg_id,'vel_median']=vel.median()
        df_test_sum.loc[seg_id,'vel_max']=vel.max()
        df_test_sum.loc[seg_id,'vel_min']=vel.min()
        df_test_sum.loc[seg_id,'vel_std']=vel.std()
        df_test_sum.loc[seg_id,'vel_var']=vel.var()
        df_test_sum.loc[seg_id,'vel_sum']=vel.sum()    
        df_test_sum.loc[seg_id,'vel_skew']=vel.skew()
        df_test_sum.loc[seg_id,'vel_kurt']=vel.kurt()

        acc = df_test.loc[seg_id, 'acc']    
        acc = pd.Series(vel)
        df_test_sum.loc[seg_id,'acc_mean']=acc.mean()
        df_test_sum.loc[seg_id,'acc_median']=acc.median()
        df_test_sum.loc[seg_id,'acc_max']=acc.max()
        df_test_sum.loc[seg_id,'acc_min']=acc.min()
        df_test_sum.loc[seg_id,'acc_std']=acc.std()
        df_test_sum.loc[seg_id,'acc_var']=acc.var()
        df_test_sum.loc[seg_id,'acc_sum']=acc.sum()    
        df_test_sum.loc[seg_id,'acc_skew']=acc.skew()
        df_test_sum.loc[seg_id,'acc_kurt']=acc.kurt()


        
        df_test_sum.loc[seg_id, 'mean_change_abs_acc'] = np.mean(np.diff(acc))
        df_test_sum.loc[seg_id, 'mean_change_abs_vel'] = np.mean(np.diff(vel))
        df_test_sum.loc[seg_id, 'mean_change_abs_disp'] = np.mean(np.diff(disp))

        df_test_sum.loc[seg_id, 'mean_change_rate_acc'] = np.mean(np.nonzero((np.diff(acc) / acc[:-1]))[0])
        df_test_sum.loc[seg_id, 'mean_change_rate_vel'] = np.mean(np.nonzero((np.diff(vel) / vel[:-1]))[0])
        df_test_sum.loc[seg_id, 'mean_change_rate_disp'] = np.mean(np.nonzero((np.diff(disp) / disp[:-1]))[0])
                
        df_test_sum.loc[seg_id, 'abs_max_acc'] = np.abs(acc).max()
        df_test_sum.loc[seg_id, 'abs_max_vel'] = np.abs(vel).max()
        df_test_sum.loc[seg_id, 'abs_max_disp'] = np.abs(disp).max()

        df_test_sum.loc[seg_id, 'abs_min_acc'] = np.abs(acc).min()
        df_test_sum.loc[seg_id, 'abs_min_vel'] = np.abs(vel).min()
        df_test_sum.loc[seg_id, 'abs_min_disp'] = np.abs(disp).min()

        df_test_sum.loc[seg_id, 'max_to_min_acc'] = acc.max() / np.abs(acc.min())
        df_test_sum.loc[seg_id, 'max_to_min_vel'] = vel.max() / np.abs(vel.min())
        df_test_sum.loc[seg_id, 'max_to_min_disp'] = disp.max() / np.abs(disp.min())

        df_test_sum.loc[seg_id, 'max_to_min_diff_acc'] = acc.max() - np.abs(acc.min())
        df_test_sum.loc[seg_id, 'max_to_min_diff_vel'] = vel.max() - np.abs(vel.min())
        df_test_sum.loc[seg_id, 'max_to_min_diff_disp'] = disp.max() - np.abs(disp.min())

        df_test_sum.loc[seg_id, 'q95_acc'] = np.quantile(acc,0.95)
        df_test_sum.loc[seg_id, 'q99_acc'] = np.quantile(acc,0.99)
        df_test_sum.loc[seg_id, 'q05_acc'] = np.quantile(acc,0.05)
        df_test_sum.loc[seg_id, 'q01_acc'] = np.quantile(acc,0.01)

        df_test_sum.loc[seg_id, 'q95_vel'] = np.quantile(vel,0.95)
        df_test_sum.loc[seg_id, 'q99_vel'] = np.quantile(vel,0.99)
        df_test_sum.loc[seg_id, 'q05_vel'] = np.quantile(vel,0.05)
        df_test_sum.loc[seg_id, 'q01_vel'] = np.quantile(vel,0.01)

        df_test_sum.loc[seg_id, 'q95_disp'] = np.quantile(disp,0.95)
        df_test_sum.loc[seg_id, 'q99_disp'] = np.quantile(disp,0.99)
        df_test_sum.loc[seg_id, 'q05_disp'] = np.quantile(disp,0.05)
        df_test_sum.loc[seg_id, 'q01_disp'] = np.quantile(disp,0.01)

        df_test_sum.loc[seg_id, 'abs_q95_acc'] = np.quantile(np.abs(acc), 0.95)
        df_test_sum.loc[seg_id, 'abs_q99_acc'] = np.quantile(np.abs(acc), 0.99)
        df_test_sum.loc[seg_id, 'abs_q05_acc'] = np.quantile(np.abs(acc), 0.05)
        df_test_sum.loc[seg_id, 'abs_q01_acc'] = np.quantile(np.abs(acc), 0.01)

        df_test_sum.loc[seg_id, 'abs_q95_vel'] = np.quantile(np.abs(vel), 0.95)
        df_test_sum.loc[seg_id, 'abs_q99_vel'] = np.quantile(np.abs(vel), 0.99)
        df_test_sum.loc[seg_id, 'abs_q05_vel'] = np.quantile(np.abs(vel), 0.05)
        df_test_sum.loc[seg_id, 'abs_q01_vel'] = np.quantile(np.abs(vel), 0.01)

        df_test_sum.loc[seg_id, 'abs_q95_disp'] = np.quantile(np.abs(disp), 0.95)
        df_test_sum.loc[seg_id, 'abs_q99_disp'] = np.quantile(np.abs(disp), 0.99)
        df_test_sum.loc[seg_id, 'abs_q05_disp'] = np.quantile(np.abs(disp), 0.05)
        df_test_sum.loc[seg_id, 'abs_q01_disp'] = np.quantile(np.abs(disp), 0.01)

        df_test_sum.loc[seg_id, 'trend_acc'] = add_trend_feature(acc)
        df_test_sum.loc[seg_id, 'abs_trend_acc'] = add_trend_feature(acc, abs_values=True)
        df_test_sum.loc[seg_id, 'abs_mean_acc'] = np.abs(acc).mean()
        df_test_sum.loc[seg_id, 'abs_std_acc'] = np.abs(acc).std()

        df_test_sum.loc[seg_id, 'trend_vel'] = add_trend_feature(vel)
        df_test_sum.loc[seg_id, 'abs_trend_vel'] = add_trend_feature(vel, abs_values=True)
        df_test_sum.loc[seg_id, 'abs_mean_vel'] = np.abs(vel).mean()
        df_test_sum.loc[seg_id, 'abs_std_vel'] = np.abs(vel).std()

        df_test_sum.loc[seg_id, 'trend_disp'] = add_trend_feature(disp)
        df_test_sum.loc[seg_id, 'abs_trend_disp'] = add_trend_feature(disp, abs_values=True)
        df_test_sum.loc[seg_id, 'abs_mean_disp'] = np.abs(disp).mean()
        df_test_sum.loc[seg_id, 'abs_std_disp'] = np.abs(disp).std()

        df_test_sum.loc[seg_id, 'Hilbert_mean_acc'] = np.abs(hilbert(acc)).mean()
        df_test_sum.loc[seg_id, 'Hann_window_mean_acc'] = (convolve(acc, hann(150), mode='same') / sum(hann(150))).mean()

        df_test_sum.loc[seg_id, 'Hilbert_mean_vel'] = np.abs(hilbert(vel)).mean()
        df_test_sum.loc[seg_id, 'Hann_window_mean_vel'] = (convolve(vel, hann(150), mode='same') / sum(hann(150))).mean()

        df_test_sum.loc[seg_id, 'Hilbert_mean_disp'] = np.abs(hilbert(disp)).mean()
        df_test_sum.loc[seg_id, 'Hann_window_mean_disp'] = (convolve(disp, hann(150), mode='same') / sum(hann(150))).mean()

        
        for w in tqdm(windows):                
            df_test_sum.loc[seg_id,f'disp_mean_mean_{w}'] = disp.rolling(window=w).mean().mean()
            df_test_sum.loc[seg_id,f'disp_mean_median_{w}'] = disp.rolling(window=w).mean().median()
            df_test_sum.loc[seg_id,f'disp_mean_max_{w}'] = disp.rolling(window=w).mean().max()
            df_test_sum.loc[seg_id,f'disp_mean_min_{w}'] = disp.rolling(window=w).mean().min()
            df_test_sum.loc[seg_id,f'disp_mean_std_{w}'] = disp.rolling(window=w).mean().std()
            df_test_sum.loc[seg_id,f'disp_mean_var_{w}'] = disp.rolling(window=w).mean().var()
            df_test_sum.loc[seg_id,f'disp_mean_sum_{w}'] = disp.rolling(window=w).mean().sum()
            df_test_sum.loc[seg_id,f'disp_mean_skew_{w}'] = disp.rolling(window=w).mean().skew()
            df_test_sum.loc[seg_id,f'disp_mean_kurt_{w}'] = disp.rolling(window=w).mean().kurt()

            df_test_sum.loc[seg_id,f'disp_std_mean_{w}'] = disp.rolling(window=w).std().mean()
            df_test_sum.loc[seg_id,f'disp_std_median_{w}'] = disp.rolling(window=w).std().median()
            df_test_sum.loc[seg_id,f'disp_std_max_{w}'] = disp.rolling(window=w).std().max()
            df_test_sum.loc[seg_id,f'disp_std_min_{w}'] = disp.rolling(window=w).std().min()
            df_test_sum.loc[seg_id,f'disp_std_std_{w}'] = disp.rolling(window=w).std().std()
            df_test_sum.loc[seg_id,f'disp_std_var_{w}'] = disp.rolling(window=w).std().var()
            df_test_sum.loc[seg_id,f'disp_std_sum_{w}'] = disp.rolling(window=w).std().sum()
            df_test_sum.loc[seg_id,f'disp_std_skew_{w}'] = disp.rolling(window=w).std().skew()
            df_test_sum.loc[seg_id,f'disp_std_kurt_{w}'] = disp.rolling(window=w).std().kurt()

        
            df_test_sum.loc[seg_id,f'vel_mean_mean_{w}'] = vel.rolling(window=w).mean().mean()
            df_test_sum.loc[seg_id,f'vel_mean_median_{w}'] = vel.rolling(window=w).mean().median()
            df_test_sum.loc[seg_id,f'vel_mean_max_{w}'] = vel.rolling(window=w).mean().max()
            df_test_sum.loc[seg_id,f'vel_mean_min_{w}'] = vel.rolling(window=w).mean().min()
            df_test_sum.loc[seg_id,f'vel_mean_std_{w}'] = vel.rolling(window=w).mean().std()
            df_test_sum.loc[seg_id,f'vel_mean_var_{w}'] = vel.rolling(window=w).mean().var()
            df_test_sum.loc[seg_id,f'vel_mean_sum_{w}'] = vel.rolling(window=w).mean().sum()
            df_test_sum.loc[seg_id,f'vel_mean_skew_{w}'] = vel.rolling(window=w).mean().skew()
            df_test_sum.loc[seg_id,f'vel_mean_kurt_{w}'] = vel.rolling(window=w).mean().kurt() 

            df_test_sum.loc[seg_id,f'vel_std_mean_{w}'] = vel.rolling(window=w).std().mean()
            df_test_sum.loc[seg_id,f'vel_std_median_{w}'] = vel.rolling(window=w).std().median()
            df_test_sum.loc[seg_id,f'vel_std_max_{w}'] = vel.rolling(window=w).std().max()
            df_test_sum.loc[seg_id,f'vel_std_min_{w}'] = vel.rolling(window=w).std().min()
            df_test_sum.loc[seg_id,f'vel_std_std_{w}'] = vel.rolling(window=w).std().std()
            df_test_sum.loc[seg_id,f'vel_std_var_{w}'] = vel.rolling(window=w).std().var()
            df_test_sum.loc[seg_id,f'vel_std_sum_{w}'] = vel.rolling(window=w).std().sum()
            df_test_sum.loc[seg_id,f'vel_std_skew_{w}'] = vel.rolling(window=w).std().skew()
            df_test_sum.loc[seg_id,f'vel_std_kurt_{w}'] = vel.rolling(window=w).std().kurt()  


            df_test_sum.loc[seg_id,f'acc_mean_mean_{w}'] = acc.rolling(window=w).mean().mean()
            df_test_sum.loc[seg_id,f'acc_mean_median_{w}'] = acc.rolling(window=w).mean().median()
            df_test_sum.loc[seg_id,f'acc_mean_max_{w}'] = acc.rolling(window=w).mean().max()
            df_test_sum.loc[seg_id,f'acc_mean_min_{w}'] = acc.rolling(window=w).mean().min()
            df_test_sum.loc[seg_id,f'acc_mean_std_{w}'] = acc.rolling(window=w).mean().std()
            df_test_sum.loc[seg_id,f'acc_mean_var_{w}'] = acc.rolling(window=w).mean().var()
            df_test_sum.loc[seg_id,f'acc_mean_sum_{w}'] = acc.rolling(window=w).mean().sum()
            df_test_sum.loc[seg_id,f'acc_mean_skew_{w}'] = acc.rolling(window=w).mean().skew()
            df_test_sum.loc[seg_id,f'acc_mean_kurt_{w}'] = acc.rolling(window=w).mean().kurt()
            
            df_test_sum.loc[seg_id,f'acc_std_mean_{w}'] = acc.rolling(window=w).std().mean()
            df_test_sum.loc[seg_id,f'acc_std_median_{w}'] = acc.rolling(window=w).std().median()
            df_test_sum.loc[seg_id,f'acc_std_max_{w}'] = acc.rolling(window=w).std().max()
            df_test_sum.loc[seg_id,f'acc_std_min_{w}'] = acc.rolling(window=w).std().min()
            df_test_sum.loc[seg_id,f'acc_std_std_{w}'] = acc.rolling(window=w).std().std()
            df_test_sum.loc[seg_id,f'acc_std_var_{w}'] = acc.rolling(window=w).std().var()
            df_test_sum.loc[seg_id,f'acc_std_sum_{w}'] = acc.rolling(window=w).std().sum()
            df_test_sum.loc[seg_id,f'acc_std_skew_{w}'] = acc.rolling(window=w).std().skew()
            df_test_sum.loc[seg_id,f'acc_std_kurt_{w}'] = acc.rolling(window=w).std().kurt() 
            
    print(df_test_sum)
    df_test_sum.to_pickle(f"{PICKLE_PATH}\\df_test_sum.pickle")
    print("df_test_sum saved")
    del df_test
else:
    df_test_sum = pd.read_pickle(f"{PICKLE_PATH}\\df_test_sum.pickle")
    print("df_test_sum loaded")


#TRAIN NETWORK
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
	
# df_test_sum = clean_dataset(df_test_sum)
# df_train_sum = clean_dataset(df_train_sum)
	
def tanh(x):
    return np.tanh(x)
def sinh(x):
    return np.sinh(x)
def cosh(x):
    return np.cosh(x)

def GeneticPrograming():   
    gp_tanh = make_function(tanh,"tanh",1)
    gp_sinh = make_function(sinh,"sinh",1)
    gp_cosh = make_function(cosh,"cosh",1)

    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    while True:
        est_gp = SymbolicRegressor(population_size=200000,
                                    tournament_size=5000,
                                    generations=10, stopping_criteria=0.0,
                                    p_crossover=0.9, p_subtree_mutation=0.001, p_hoist_mutation=0.001, p_point_mutation=0.001,
                                    max_samples=1.0, verbose=1,
                                    function_set = ('add', 'sub', 'mul', 'div', gp_tanh, 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),
                                    #function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),
                                    metric = 'mean absolute error', warm_start=True,
                                    n_jobs = 1, parsimony_coefficient=0.001, random_state=11)

        if (os.path.exists(f'{PICKLE_PATH}\\EQS_gp.pickle')):
            pickle_in = open(f'{PICKLE_PATH}\\EQS_gp.pickle', 'rb')
            est_gp = pickle.load(pickle_in)
            print("Model Loaded")

        est_gp.generations += 10
        est_gp.p_subtree_mutation /= 10
        est_gp.p_hoist_mutation /= 10
        est_gp.p_point_mutation /= 10
        est_gp.parsimony_coefficient /= 10

        alldata = pd.concat([X_tr, X_test])
        scaler = StandardScaler()
        alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

        X_tr_scaled = alldata[:X_tr.shape[0]]
        X_test_scaled = alldata[X_tr.shape[0]:]

        est_gp.fit(X_tr_scaled, y_tr)

        with open(f'{PICKLE_PATH}\\EQS_gp.pickle','wb') as f:
            pickle.dump(est_gp, f)
            print('Model Saved')

        y_gp = est_gp.predict(X_tr_scaled)
        gpLearn_MAE = mean_absolute_error(y_tr, y_gp)
        print("gpLearn MAE:", gpLearn_MAE)

        submission.time_to_failure = est_gp.predict(X_test_scaled)
        submission.to_csv(f'{DATA_PATH}\\gplearnEQS_submission.csv', index=True)
        print(submission.head())

def LGB():
    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    alldata = pd.concat([X_tr, X_test])
    scaler = StandardScaler()
    alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

    X_tr_scaled = alldata[:X_tr.shape[0]]
    X_test_scaled = alldata[X_tr.shape[0]:]

    gbm = lgb.LGBMRegressor(boosting_type='gbdt',
                            objective='huber',
                            colsample_bytree=1,
                            learning_rate=0.01,                            
                            verbosity=-1,
                            max_depth=-1,
                            n_estimators=200_000,
                            n_jobs=2,
                            metric='mae')

    predictions = np.zeros(len(X_test))
    prediction_count = 0
    n_fold = 50
    folds = KFold(n_splits = n_fold, shuffle = True, random_state = 101)
    for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X_tr_scaled))):
        print('')
        print("---------------------------------------------------")
        print('Fold', fold_n)
        print("---------------------------------------------------")

        X_train_f, X_valid = X_tr_scaled.iloc[train_index], X_tr_scaled.iloc[valid_index]
        y_train_f, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]

        gbm.fit(X_train_f, y_train_f.values.flatten(),
                eval_set=[(X_valid, y_valid.values.flatten())],
                eval_metric = 'mae', verbose=1000, early_stopping_rounds = 200)
        
        y_pred_valid = gbm.predict(X_valid, num_iteration = gbm.best_iteration_)
        y_pred_all = gbm.predict(X_tr_scaled, num_iteration = gbm.best_iteration_)        
        valid_mae = mean_absolute_error(y_valid.values.flatten(), y_pred_valid)
        mae = mean_absolute_error(y_tr.values.flatten(), y_pred_all)
        
        print('The valid mae of prediction is:', valid_mae)
        print('The mae of prediction is:', mae)        
        print('')

        if mae < 2.0 and valid_mae < 2.2:
            print("Good MAE for col remove")
            prediction = gbm.predict(X_test_scaled)

            if abs(mae - valid_mae) < 1:
                print("Good MAE for Predictions")
                predictions += prediction
                prediction_count += 1

            df_importance = pd.DataFrame(data = {'importance': gbm.feature_importances_}, index=X_valid.columns)
            df_importance.sort_values(by = 'importance', axis = 0, inplace = True, ascending=False)
            print("Importance")
            print(df_importance.head())
            
            remove_cols = df_importance[-30:]
            remove_cols = list(remove_cols[remove_cols<=0].dropna().index.values)
            if len(remove_cols) > 0:
                print("Remove Cols:", remove_cols)        
                X_tr_scaled.drop(inplace = True, columns = remove_cols)
                X_test_scaled.drop(inplace = True, columns = remove_cols)
                print("New Col Count:", len(X_tr_scaled.columns))                
        
    submission.time_to_failure = predictions / prediction_count     

    submission.to_csv(f'{DATA_PATH}\\LGB_EQS_v4_submission.csv', index=True)
    print(submission.head())
    print(submission.tail())
    print(submission.describe())

def XGB():
    xgb_params = {'eta': 0.01,
              'max_depth': 20,
              'subsample': 0.5,
              'objective': 'reg:linear',
              'grow_policy': 'lossguide',
              'eval_metric': 'mae',     
              'process_type':'default',
              'tree_method':'exact',
              'silent':True,
              'nthread': 4}

    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    alldata = pd.concat([X_tr, X_test])
    scaler = StandardScaler()
    alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

    X_tr_scaled = alldata[:X_tr.shape[0]]
    X_test_scaled = alldata[X_tr.shape[0]:]

    predictions = np.zeros(len(X_test))
    prediction_count = 0
    n_fold = 50
    folds = KFold(n_splits = n_fold, shuffle = True, random_state = 101)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_tr_scaled)):
        
        print('Fold', fold_n)
        print("-------------------------------------------------------")

        X_train_f, X_valid = X_tr_scaled.iloc[train_index], X_tr_scaled.iloc[valid_index]
        y_train_f, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]

        train_data = xgb.DMatrix(data=X_train_f, label=y_train_f, feature_names=X_train_f.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_valid.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, 
                          num_boost_round=200000, evals=watchlist, early_stopping_rounds=200, 
                          verbose_eval=500,
                          params=xgb_params)
        
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_valid.columns), ntree_limit=model.best_ntree_limit)        
        y_pred_mae_valid = mean_absolute_error(y_valid.values.flatten(), y_pred_valid)
        print('The mae of valid prediction is:', y_pred_mae_valid)

        y_pred_train = model.predict(xgb.DMatrix(X_tr_scaled, feature_names=X_tr_scaled.columns), ntree_limit=model.best_ntree_limit)        
        y_pred_mae = mean_absolute_error(y_tr.values.flatten(), y_pred_train)
        print('The mae of prediction is:', y_pred_mae)

        if y_pred_mae < 2 and y_pred_mae_valid < 2:
            print("Good Predictions, Included")
            y_pred = model.predict(xgb.DMatrix(X_test_scaled, feature_names=X_test_scaled.columns), ntree_limit=model.best_ntree_limit)
            predictions += y_pred
            prediction_count += 1

            print("prediction_count", prediction_count)
            print("Prediction", predictions/prediction_count)
            print("-------------------------------------------------------")
        else:
            print("Bad Predictions, Discarded")

    submission.time_to_failure = predictions/prediction_count        
    submission.to_csv(f'{DATA_PATH}\\XGB_EQS_v2_submission.csv', index=True)
    print(submission.head())
    print(submission.tail())
    print(submission.describe())

def KERAS():
    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    alldata = pd.concat([X_tr, X_test])
    scaler = MaxAbsScaler()
    alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

    X_tr_scaled = alldata[:X_tr.shape[0]]
    X_test_scaled = alldata[X_tr.shape[0]:]

    X = np.array(X_tr_scaled)
    y = np.array(y_tr)

    n_features = len(X_tr_scaled.columns)

    BATCH_SIZE = 1000
    EPOCHS = 1000
    modelName = f"Keras_EQS_v15"

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
        model.add(Dense(n_features, input_dim=n_features, activation='relu'))            
        
        model.add(Dense(512, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))        

        model.add(Dense(256, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(128, kernel_initializer='normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.25))

        model.add(Dense(1, kernel_initializer='normal'))    
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='sgd', metrics=['mae'])
        model.fit(X, y, validation_split = 0.5, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)
        model.save(f"{MODEL_PATH}\\{model.name}.h5")
        print("Model Saved.", modelName)

    print(model.summary())

    print("Predicting with Model", modelName)
    X_pred = np.array(X_test_scaled)
    y_pred = model.predict(X_pred)
    print("Prediction Completed")

    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
    submission['time_to_failure'] = y_pred
    print(submission.head())
    submission.to_csv(f'{DATA_PATH}\\submission{modelName}.csv')
    print("Submission File Created")

def CATBOOST():	
	
    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    alldata = pd.concat([X_tr, X_test])
    scaler = StandardScaler()
    alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

    X_tr_scaled = alldata[:X_tr.shape[0]]
    X_test_scaled = alldata[X_tr.shape[0]:]

    model = CatBoostRegressor(iterations = 20000, use_best_model = True, verbose = 100, loss_function= 'MAE', thread_count = 4)

    predictions = np.zeros(len(X_test))
    prediction_count = 0
    n_fold = 50
    folds = KFold(n_splits = n_fold, shuffle = True, random_state = 101)    
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_tr_scaled)):
        print('Fold', fold_n)
        print("-------------------------------------------------------")

        X_train_f, X_valid = X_tr_scaled.iloc[train_index], X_tr_scaled.iloc[valid_index]
        y_train_f, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]
        model.fit(X_train_f, y_train_f, eval_set=(X_valid, y_valid), early_stopping_rounds=1000)

        y_pred_valid = model.predict(X_valid)
        y_pred_all = model.predict(X_tr_scaled)        
        valid_mae = mean_absolute_error(y_valid.values.flatten(), y_pred_valid)
        mae = mean_absolute_error(y_tr.values.flatten(), y_pred_all)
        print("mae: ",mae)
        print("valid_mae: ",valid_mae)

        if mae < 2.0 and valid_mae < 2.2:
            print("Good MAE for prediction")
            prediction = model.predict(X_test_scaled)
            predictions += prediction
            prediction_count += 1
            print(predictions / prediction_count)

    submission.time_to_failure = predictions / prediction_count
    submission.to_csv(f'{DATA_PATH}\\CATBOOST_EQS_v2_submission.csv', index=True)
    print(submission.head())
    print(submission.tail())
    print(submission.describe())

def RANDOMFOREST():
    X_test = df_test_sum.drop('time', axis=1).fillna(0)
    X_tr = df_train_sum.drop('time',  axis=1).fillna(0)
    y_tr = df_train_sum['time']

    alldata = pd.concat([X_tr, X_test])
    scaler = StandardScaler()
    alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

    X_tr_scaled = alldata[:X_tr.shape[0]]
    X_test_scaled = alldata[X_tr.shape[0]:]

    model = RandomForestRegressor(n_estimators = 0,
                                  criterion = 'mae',   
                                  n_jobs = 2,
                                  verbose = 0,
                                  warm_start = True)

    n_fold = 50
    folds = KFold(n_splits = n_fold, shuffle = True, random_state = 101)
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_tr_scaled)):
        print('Fold', fold_n)
        print("-------------------------------------------------------")
        model.n_estimators += 10
        print('n_estimators', model.n_estimators)

        X_train_f, X_valid = X_tr_scaled.iloc[train_index], X_tr_scaled.iloc[valid_index]
        y_train_f, y_valid = y_tr.iloc[train_index], y_tr.iloc[valid_index]
        model.fit(X_train_f, y_train_f)

        y_pred_valid = model.predict(X_valid)
        y_pred_all = model.predict(X_tr_scaled)        
        valid_mae = mean_absolute_error(y_valid.values.flatten(), y_pred_valid)
        mae = mean_absolute_error(y_tr.values.flatten(), y_pred_all)
        print("mae: ",mae)
        print("valid_mae: ",valid_mae)

    prediction = model.predict(X_test_scaled)
    submission.time_to_failure = prediction
    submission.to_csv(f'{DATA_PATH}\\RANDOMFOREST_EQS_v1_submission.csv', index=True)
    print(submission.head())
    print(submission.tail())
    print(submission.describe())
        
#LGB()
#XGB()
#GeneticPrograming()
#KERAS()
#CATBOOST()
RANDOMFOREST()
   