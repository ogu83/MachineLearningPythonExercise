import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection, neighbors, svm

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


EUR_USD_FILE = "EUR_USD Historical Data.csv"
MOVING_AVERAGE_WINDOWS = 100

df = pd.DataFrame()
df = pd.read_csv(EUR_USD_FILE)
df.drop("Change", inplace=True, axis=1)
df = df[::-1] #reverse

df['PriceF1'] = df['Price'].shift(-1)
df['Buy'] = df['PriceF1'] // df['Price']
#df.drop(['Price','Open','High','Low'], inplace=True, axis=1)

for i in range(1, MOVING_AVERAGE_WINDOWS):
    df[f"MA{i}_P"] = df["Price"].rolling(window=i).mean()
    df[f"MA{i}_O"] = df["Open"].rolling(window=i).mean()
    df[f"MA{i}_H"] = df["High"].rolling(window=i).mean()
    df[f"MA{i}_L"] = df["Low"].rolling(window=i).mean()

df = df[(MOVING_AVERAGE_WINDOWS-2):]
df = df[:-1]
df.fillna(value=0, axis=1, inplace=True)

#print(df.head())
#print(df.tail())
#print(df.describe())

#plt.plot(df["Price"])
#plt.plot(df[f"MA9_P"])
#plt.plot(df[f"MA9_L"])
#plt.plot(df[f"MA9_H"])
#plt.show()

#y_tr = df["Buy"]
#X_tr = df.drop(["PriceF1", "Buy", "Date"], axis=1)

#Preprocessing Data
X = np.array(df.drop(["PriceF1", "Buy", "Date"],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Buy'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#Classifiers

#Nu Support Vector Classifier - Failed with 0.51 Accuracy
#clf = svm.NuSVC(random_state=11, verbose=True) 

#Support Vector Classifier - Failed with 0.48 Accuracy
#clf = svm.SVC(random_state=11, verbose=True) 

#Random Forest Reggressor
clf = RandomForestRegressor(verbose=1,random_state=11,n_jobs=-1,n_estimators=100)

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("accuracy: ", accuracy)