import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

quandl.ApiConfig.api_key = "rVUhNwziaabf9P-EFxZ-"

df = quandl.get_table('FXCM/D1', symbol='EUR/USD')
#df.set_index('date')

print(df.head())
#last_date = df.iloc[-1].date
#print(last_date)

def FormatForModel(dA):
        dA = dA[['openbid', 'highbid', 'lowbid', 'closebid']]
        dA['HL_PCT'] = (dA['highbid'] - dA['closebid']) / dA['closebid'] * 100.0
        dA['PCT_change'] = (dA['closebid'] - dA['openbid']) / dA['openbid'] * 100.0
        dA = dA[['closebid', 'HL_PCT', 'PCT_change']]
        return dA
        
df = FormatForModel(df)

forecast_col = "closebid"
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(["label"],1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df["label"])
#y = np.array(df["label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2)

clf = LinearRegression()
#clf = svm.SVR(kernel="poly")
#clf = svm.SVR()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("Accuracy")
print(accuracy)

forecast_set = clf.predict(X_lately)
print("Forecast Set")
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

#last_date = df.iloc[-1].name
#last_unix = last_date.timestamp()
#one_day = 86400
#next_unix = last_unix + one_day

#for i in forecast_set:
#    next_date = datetime.datetime.fromtimestamp(next_unix)
#    next_unix += one_day
#    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

last_date = df.iloc[-1].name
last_unix = last_date
one_day = 1
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = next_unix
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
   
print(df)
        
df[forecast_col].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#print(df.head())
#print(df.tail())