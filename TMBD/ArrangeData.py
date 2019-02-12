import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
#%matplotlib inline
from tqdm import tqdm
import json, ast

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
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge

def getUniqueIdFromString(value):
    return int(abs(hash(value)) % (10 ** 8))

def getUniqueIdFromStringArray(values):
    return [getUniqueIdFromString(x) for x in values]

def getIdListFromJson(data):
    datas = data.values.flatten()
    ids = []
    for c in tqdm(datas):    
        ccc = []
        if isNaN(c) == False:
            c = json.dumps(ast.literal_eval(c))        
            c = json.loads(c)            
            for cc in c:
                ccInt = int(cc["id"])
                ccc.append(ccInt)
        else:
            ccc.append(0)
        ids.append(ccc)
    return ids

def getIsoListFormJson(data, isoKey):
    datas = data.values.flatten()
    ids = []
    for c in tqdm(datas):    
        ccc = []
        if isNaN(c) == False:
            c = json.dumps(ast.literal_eval(c))        
            c = json.loads(c)            
            for cc in c:
                ccInt = getUniqueIdFromString(cc[isoKey])
                ccc.append(ccInt)
        else:
            ccc.append(0)
        ids.append(ccc)
    return ids

def pushCategoryDataAsColumns(data, ids, colName):
    print(colName)
    for id in tqdm(range(len(ids))):
        for idd in tqdm(ids[id]):
            fullColName = colName + "_" + str(idd)
            if fullColName not in data.columns:
                 data[fullColName] = np.zeros(len(ids))
            data[fullColName][id] = 1

def isNaN(x):
    return str(x) == str(1e400*0)

def arrangeData(data):    
    retval = pd.DataFrame()    
    
    print("belongs_to_collection_ids")
    belongs_to_collection_ids = getIdListFromJson(data["belongs_to_collection"])
    print("or_lang_ids")
    or_lang_ids = getUniqueIdFromStringArray(data["original_language"])
    print("genre_ids")
    genre_ids = getIdListFromJson(data["genres"])
    print("prod_comp_ids")
    prod_comp_ids = getIdListFromJson(data["production_companies"])
    print("prod_coutry_ids")
    prod_coutry_ids = getIsoListFormJson(data["production_countries"],"iso_3166_1")
    print("spoken_language_ids")
    spoken_language_ids = getIsoListFormJson(data["spoken_languages"],"iso_639_1")
    print("status_ids")
    status_ids = getUniqueIdFromStringArray(data["status"])
    print("keyword_ids")
    keyword_ids = getIdListFromJson(data["Keywords"])
    print("cast_ids")
    cast_ids = getIdListFromJson(data["cast"])
    print("crew_ids")
    crew_ids = getIdListFromJson(data["crew"])

    #retval["id"] = data["id"]
    
    #retval["collection"] = belongs_to_collection_ids
    pushCategoryDataAsColumns(retval, belongs_to_collection_ids, "collection")

    #retval["genre"] = genre_ids
    pushCategoryDataAsColumns(retval, genre_ids, "genre")

    retval["budget"] = data["budget"]
    
    retval["or_lang"] = or_lang_ids
    #pushCategoryDataAsColumns(retval, or_lang_ids, "or_lang")

    retval["popularity"] = data["popularity"]
    
    #retval["prod_comp"] = prod_comp_ids
    pushCategoryDataAsColumns(retval, prod_comp_ids, "prod_comp")

    #retval["prod_country"] = prod_coutry_ids
    pushCategoryDataAsColumns(retval, prod_coutry_ids, "prod_country")

    retval["runtime"] = data["runtime"]
    
    #retval["s_lang"] = spoken_language_ids
    pushCategoryDataAsColumns(retval, spoken_language_ids, "s_lang")

    retval["stat"] = status_ids
    #pushCategoryDataAsColumns(retval, status_ids, "stat")

    #retval["keywords"] = keyword_ids
    pushCategoryDataAsColumns(retval, keyword_ids, "keywords")

    #retval["cast"] = cast_ids
    pushCategoryDataAsColumns(retval, cast_ids, "cast")

    #retval["crew"] = crew_ids
    pushCategoryDataAsColumns(retval, crew_ids, "crew")

    return retval;

TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
LABEL_COL_NAME = "revenue"
TR_X_PICKLE = "./tr_x.pickle"
TR_Y_PICKLE = "./tr_y.pickle"
TEST_X_PICKLE = "./test_x.pickle"

if (os.path.exists(TR_X_PICKLE) == False):
    train = pd.read_csv(TRAIN_DATA_PATH)
    tr_X = arrangeData(train)
    tr_y = train[LABEL_COL_NAME]
    tr_X.to_pickle(TR_X_PICKLE)
    tr_y.to_pickle(TR_Y_PICKLE)
else:
    tr_X = pd.read_pickle(TR_X_PICKLE)
    tr_y = pd.read_pickle(TR_Y_PICKLE)

print("Trainin Data")
#print(tr_X.head())
print(tr_X.describe())
#print(tr_y.head())

if (os.path.exists(TEST_X_PICKLE) == False):
    test = pd.read_csv(TEST_DATA_PATH)
    test_X = arrangeData(test)
else:
    test_X = pd.read_pickle(TEST_X_PICKLE)

print("Test Data")
#print(test_X)
print(test_X.describe())

X_train, X_test, y_train, y_test = model_selection.train_test_split(tr_X,tr_y,test_size=0.2)

rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                          max_features=0.5, #Max number of features each tree can use 
                          min_samples_leaf=30, #Min amount of samples in each leaf
                          random_state=11)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print("Accuracy:", accuracy)

predictions = rf.predict(tr_X)
print("First 10 Labels")
print(tr_y[10:])
print("Predictions Labels for first 10 data")
print(predictions[10:])
print("Mean Absolute Error")
print(mean_absolute_error(tr_y, predictions))
print("Mean Squared Error")
print(mean_squared_error(tr_y, predictions))