import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


EUR_USD_FILE = "EUR_USD Historical Data.csv"
MOVING_AVERAGE_WINDOWS = 366

df = pd.DataFrame()
df = pd.read_csv(EUR_USD_FILE)
df.drop("Change", inplace=True, axis=1)
df = df[::-1] #reverse

#df['PriceF1'] = df['Price'].shift(-1)
#df['Buy'] = df['PriceF1'] // df['Price']
#df.drop(['Price','Open','High','Low'], inplace=True, axis=1)

for i in [3,5,9,14,29,50,100,150,300,450]:
    
    price_roll_std = df["Price"].rolling(window=i).std()
    price_roll_mean = df["Price"].rolling(window=i).mean()
    
    price_q25 = df["Price"].rolling(window=i).quantile(0.25)
    price_q75 = df["Price"].rolling(window=i).quantile(0.75)
    price_q50 = df["Price"].rolling(window=i).quantile(0.50)

    #Quantiles
    df[f"Q25{i}_P"] = price_q25
    df[f"Q75{i}_P"] = price_q75
    df[f"Q50{i}_P"] = price_q50

    #Moving Averages
    df[f"MA{i}_P"] = price_roll_mean
    df[f"MA{i}_O"] = df["Open"].rolling(window=i).mean()
    df[f"MA{i}_H"] = df["High"].rolling(window=i).mean()
    df[f"MA{i}_L"] = df["Low"].rolling(window=i).mean()
    #Std
    df[f"STD{i}_P"] = price_roll_std
    df[f"STD{i}_O"] = df["Open"].rolling(window=i).std()
    df[f"STD{i}_H"] = df["High"].rolling(window=i).std()
    df[f"STD{i}_L"] = df["Low"].rolling(window=i).std()
    #Min
    df[f"MIN{i}_P"] = df["Price"].rolling(window=i).min()
    df[f"MIN{i}_O"] = df["Open"].rolling(window=i).min()
    df[f"MIN{i}_H"] = df["High"].rolling(window=i).min()
    df[f"MIN{i}_L"] = df["Low"].rolling(window=i).min()    
    #Max
    df[f"MAX{i}_P"] = df["Price"].rolling(window=i).max()
    df[f"MAX{i}_O"] = df["Open"].rolling(window=i).max()
    df[f"MAX{i}_H"] = df["High"].rolling(window=i).max()
    df[f"MAX{i}_L"] = df["Low"].rolling(window=i).max()
    #Skew
    df[f"SKEW{i}_P"] = df["Price"].rolling(window=i).skew()
    df[f"SKEW{i}_O"] = df["Open"].rolling(window=i).skew()
    df[f"SKEW{i}_H"] = df["High"].rolling(window=i).skew()
    df[f"SKEW{i}_L"] = df["Low"].rolling(window=i).skew()
    #Kurt
    df[f"SKEW{i}_P"] = df["Price"].rolling(window=i).kurt()
    df[f"SKEW{i}_O"] = df["Open"].rolling(window=i).kurt()
    df[f"SKEW{i}_H"] = df["High"].rolling(window=i).kurt()
    df[f"SKEW{i}_L"] = df["Low"].rolling(window=i).kurt()

#df = df[(MOVING_AVERAGE_WINDOWS-2):]
df = df[:-1]
df.fillna(value=0, axis=1, inplace=True)

print(df.head())
print(df.tail())
print(df.describe())

#plt.plot(df["Price"])
#plt.plot(df[f"MA9_P"])
#plt.plot(df[f"MA9_O"])
#plt.plot(df[f"MA9_L"])
#plt.plot(df[f"MA9_H"])

#plt.plot(df[f"STD9_P"])
#plt.plot(df[f"STD9_O"])
#plt.plot(df[f"STD9_L"])
#plt.plot(df[f"STD9_H"])

#plt.plot(df[f"MIN9_P"])
#plt.plot(df[f"MIN9_O"])
#plt.plot(df[f"MIN9_L"])
#plt.plot(df[f"MIN9_H"])

#plt.plot(df[f"MAX9_P"])
#plt.plot(df[f"MAX9_O"])
#plt.plot(df[f"MAX9_L"])
#plt.plot(df[f"MAX9_H"])

#plt.show()

y_tr = df["Price"]
X_tr = df.drop(["Price", "High", "Low","Date"], axis=1)

#print(X_tr)
#print(y_tr)

##TRAINING AND PREDICTING WITH GPLEARN
def tanh(x):
    return np.tanh(x);
def sinh(x):
    return np.sinh(x);
def cosh(x):
    return np.cosh(x);

gp_tanh = make_function(tanh,"tanh",1)
gp_sinh = make_function(sinh,"sinh",1)
gp_cosh = make_function(cosh,"cosh",1)

est_gp = SymbolicRegressor(population_size=20000,
                            tournament_size=500,
                            generations=1, stopping_criteria=0.0,
                            p_crossover=0.9, p_subtree_mutation=0.0001, p_hoist_mutation=0.0001, p_point_mutation=0.0001,
                            max_samples=1.0, verbose=1,
                            function_set = ('add', 'sub', 'mul', 'div', gp_tanh, 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),
                            #function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),
                            metric = 'mean absolute error', warm_start=True,
                            n_jobs = 1, parsimony_coefficient=0.0001, random_state=111)

if (os.path.exists(f'est_gp.pickle')):
    pickle_in = open(f'est_gp.pickle','rb')
    est_gp = pickle.load(pickle_in)
    print("Model Loaded")

est_gp.generations = est_gp.generations + 9
est_gp.fit(X_tr, y_tr)

with open(f'est_gp.pickle','wb') as f:
    pickle.dump(est_gp, f)
    print('Model Saved')

print("gpLearn Program:", est_gp._program)
y_gp = est_gp.predict(X_tr)
gpLearn_MAE = mean_absolute_error(y_tr, y_gp)
print("gpLearn MAE:", gpLearn_MAE)
print(y_tr, y_gp)
