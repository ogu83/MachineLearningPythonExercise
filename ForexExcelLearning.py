import os
import pandas as pd
import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

def FormatForModel(dA):
        dA = dA[['Price2', 'HL_PCT', 'PCT_change', 'RSI14', 'BIST100', 'BIST30', 'DOW30', 'DB_PCT', 'BIST_IND', 'DJI', 'BI_DJI_PCT', 'EURTRY', 'ElectionDate', 'DJTITANS']]
        return dA

# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
#print(cwd)
# Change directory 
os.chdir("/media/sf_F_DRIVE")
# List all files and directories in current directory
files = os.listdir('.')
#print(files)
# Assign spreadsheet filename to `file`
file = 'ForexForecast1.xlsx'
xl = pd.ExcelFile(file)
#print(xl.sheet_names)
# Load a sheet into a DataFrame by name: df1
df = xl.parse('USDTRY')
df = FormatForModel(df)
#df = df[::-1]       
#df = df.reindex(in dex=df.index[::-1])
df = df.iloc[::-1].reset_index(drop=True)
#print(df.head())
#print(df.tail())

forecast_col = "Price2"
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.003*len(df)))
df["label"] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(["label"],1))
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df["label"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.2)

clf = LinearRegression(n_jobs=-1)
#kernels can be linear,poly,rbf,sigmoid,precomputed
#clf = svm.SVR(kernel="poly",degree=7)
#clf = svm.SVR(kernel="rbf")

clf.fit(X_train, y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print("Accuracy")
print(accuracy)

#print(df)

forecast_set = clf.predict(X_lately)
print("Forecast Set")
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date
one_day = 1
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = next_unix
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
   
print(df)
        
df['label'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()