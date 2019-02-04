from keras.models import Sequential
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

import numpy as np
import pandas as pd
import os

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"

#Some Random Data will be changed with actual data afterwards
#data = np.random.random((1000,100))
#labels = np.random.random((1000,1))

X_train_scaled = np.nan_to_num(np.load(NP_DATA_PATH + "\\X_train_scaled.npy"))
X_test_scaled = np.nan_to_num(np.load(NP_DATA_PATH + "\\X_test_scaled.npy"))
y_tr = np.nan_to_num(np.load(NP_DATA_PATH + "\\y_tr.npy"))

data = X_train_scaled
labels = y_tr

print(data)
print(labels)

#to create a model
#model = Sequential()
#model.add(LSTM(128, input_shape=(X_train_scaled.shape[1:]), activation='relu', return_sequences=True))
#model.add(Dense(1))
#model.compile(optimizer = 'adam', loss = 'mse', metrics=['mae'])
#model.fit(data, labels, epochs=10, batch_size=10)

#predictions = model.predict(data[:5])

#TRAIN A MODEL
X_train, X_test, y_train, y_test = model_selection.train_test_split(data,labels,test_size=0.2)
#rf = svm.SVR(kernel='poly', gamma='scale', C=1.0, tol=1e-6)
#rf = svm.LinearSVR()

#rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
#                          max_features=0.5, #Max number of features each tree can use 
#                          min_samples_leaf=30, #Min amount of samples in each leaf
#                          random_state=11)

#rf = DecisionTreeRegressor(criterion='mse', max_features=0.5, min_samples_leaf=30, random_state=11)
#rf = DecisionTreeRegressor()

#rf = ExtraTreeRegressor(criterion='mse', max_features=0.5, min_samples_leaf=30, random_state=11)

#rf = BaggingRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
#                      max_features=0.5, #Max number of features each tree can use                           
#                      random_state=11)
#rf = BaggingRegressor()
#rf.fit(X_train, y_train)
#LOOK FOR ACCURACY AND ERROR
#accuracy = rf.score(X_test, y_test)

from xgboost import XGBRegressor
rf = XGBRegressor(n_estimators=100, learning_rate=.05, silent=True).fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)],eval_metric='mae', verbose=False)

predictions = rf.predict(data[10:])
print("First 10 Labels")
print(labels[10:])
print("Predictions Labels for first 10 data")
print(predictions)
print("Mean Absolute Error")
print(mean_absolute_error(labels, rf.predict(data)))
#print("Accuracy")
#print(accuracy)

#PREDCIT THE TEST DATA
test_predictions = rf.predict(X_test_scaled)

#MAKE SUBMISSION FILE
submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
submission['time_to_failure'] = test_predictions
print(submission.head())
submission.to_csv(f'{DATA_PATH}\\submission.csv') 
