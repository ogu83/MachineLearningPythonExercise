import numpy as np
import pandas as pd
import os

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

from tqdm import tqdm
import json, ast


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


def isNaN(x):
    return str(x) == str(1e400*0)


TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
LABEL_COL_NAME = "revenue"
TAKE_COL_NAMES = ["budget","popularity","runtime"]
WITH_PLOT = False

train_main = pd.read_csv(TRAIN_DATA_PATH)

drop_cols = list(filter(lambda c: c not in TAKE_COL_NAMES and c != LABEL_COL_NAME, train_main.columns))
train = train_main.drop(columns=drop_cols)
train.fillna(value=0.0, inplace = True) 

cast_ids = getIdListFromJson(train_main["cast"])
unique_cast_ids = cast_ids
crew_ids = getIdListFromJson(train_main["crew"])

train["Has_HomePage"] = list(map(lambda c: float(c is not np.nan), train_main["homepage"]))
train["CastCount"] = list(map(lambda c: float(len(c)), cast_ids))
train["CrewCount"] = list(map(lambda c: float(len(c)), crew_ids))
train["IsReleased"] = list(map(lambda c: float(c == "Released"), train_main["status"]))

print(train.describe())

train_X = train.drop([LABEL_COL_NAME], 1)
train_Y = train[LABEL_COL_NAME]

if WITH_PLOT:
    for feature in train_X.columns:
        plt.scatter(train_X[feature], train_Y)
        plt.xlabel(feature)
        plt.ylabel(LABEL_COL_NAME)
        plt.show()

X=np.array(train_X)
y=np.array(train_Y)
#X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X,y,test_size=0.2)

#print(X_train)
#print(y_train)

#rf = svm.SVR()
#rf.fit(X_train,y_train)
#rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
#                          max_features=0.5, #Max number of features each tree can use 
#                          min_samples_leaf=30, #Min amount of samples in each leaf
#                          random_state=11)
#rf.fit(X_train, y_train)
#accuracy = rf.score(X_test, y_test)
#print("Accuracy:", accuracy)

#predictions = rf.predict(X)
#print("First 10 Labels")
#print(y[10:])
#print("Predictions Labels for first 10 data")
#print(predictions[10:])
#print("Mean Absolute Error")
#print(mean_absolute_error(y, predictions))
#print("Mean Squared Error")
#print(mean_squared_error(y, predictions))

#params = {
#    'num_leaves': 54,
#    'min_data_in_leaf': 79,
#    'objective': 'huber',
#    'max_depth': -1,
#    'learning_rate': 0.001,
#    "boosting": "gbdt",
#    "bagging_freq": 5,
#    "bagging_fraction": 0.8126672064208567,
#    "bagging_seed": 11,
#    "metric": 'mse',
#    "verbosity": -1,
#    'reg_alpha': 0.1302650970728192,
#    'reg_lambda': 0.3603427518866501
#}
#rf = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
#rf.fit(X_train, y_train, 
#        eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mse',
#        verbose=10000, early_stopping_rounds=200)
            
#y_pred_valid = rf.predict(X_valid)
#print("LGB Mean Squared Error")
#print(mean_squared_error(y, y_pred_valid))

#y_pred = rf.predict(X_test, num_iteration=model.best_iteration_)

EPOCHS = 100
BATCH_SIZE = 50
modelName="v6"
cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                            histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

model = Sequential(name=modelName)
model.add(Dense(6, input_dim=len(train_X.columns), activation='relu'))
model.add(Dense(6))
model.add(Dense(6))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(X, y, validation_split = 0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)