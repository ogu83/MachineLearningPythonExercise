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
MOVING_AVERAGE_WINDOWS = 9

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

print(df.head())
print(df.tail())
print(df.describe())

#plt.plot(df["Price"])
#plt.plot(df[f"MA9_P"])
#plt.plot(df[f"MA9_L"])
#plt.plot(df[f"MA9_H"])
#plt.show()

y_tr = df["Buy"]
X_tr = df.drop(["PriceF1","Buy", "Date"], axis=1)

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

est_gp = SymbolicRegressor(population_size=2000,
                            tournament_size=50,
                            generations=1, stopping_criteria=0.0,
                            p_crossover=0.9, p_subtree_mutation=0.001, p_hoist_mutation=0.001, p_point_mutation=0.001,
                            max_samples=1.0, verbose=1,
                            function_set = ('add', 'sub', 'mul', 'div', gp_tanh, 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),
                            #function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),
                            metric = 'mean absolute error', warm_start=True,
                            n_jobs = 1, parsimony_coefficient=0.001, random_state=11)

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
