import pandas as pd
pd.options.mode.use_inf_as_na = True

import numpy as np
from numpy import diff

import os



DATA_PATH = "D:\\LANLEarthquakeData"
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"
MODEL_PATH = f"{DATA_PATH}\\models"

if os.path.exists(f'{PICKLE_PATH}\\quake_df_all.pickle'):
    train_df = pd.read_pickle(f'{PICKLE_PATH}\\quake_df_all.pickle')    
    print(train_df.drop(train_df.columns[0],inplace=True))
    train_df.to_csv(f'{DATA_PATH}\\quake_df_all.csv')
    print("quake_df_all loaded")   

#if os.path.exists(f'{PICKLE_PATH}\\quake_df_test_all.pickle'):
#    test_df = pd.read_pickle(f'{PICKLE_PATH}\\quake_df_test_all.pickle')
#    test_df.to_csv(f'{DATA_PATH}\\quake_df_test_all.csv')
#    print("test_df_all loaded")  