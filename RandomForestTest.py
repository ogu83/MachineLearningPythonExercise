import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, GRU
from keras.optimizers import adam

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from gplearn.genetic import SymbolicRegressor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

#FORMULA = LOG(MIN/MAX+AVERAGE)
def f(a):
    return 100 * (np.min(a)/np.max(a) + np.average(a))
    
X = np.random.rand(5000, 4)
y = np.array([f(x) for x in X])

X_test = np.random.rand(2000, 4)
y_test = np.array([f(x) for x in X_test])

alldata = pd.concat([pd.DataFrame(X), pd.DataFrame(X_test)])
scaler = StandardScaler()
alldata = pd.DataFrame(scaler.fit_transform(alldata), columns=alldata.columns)

X = alldata[:X.shape[0]]
X_test = alldata[X.shape[0]:]

#model = Sequential()
###model.add(GRU(48, activation='tanh', input_shape=(None, n_features)))
#model.add(Dense(4, activation='tanh', input_dim=4))
##model.add(Dense(512, activation='relu'))
##model.add(Dense(128, activation='relu'))
##model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='linear'))
#model.summary()

##model.compile(optimizer=adam(lr=0.0005), loss="mse")
#model.compile(loss='mae', optimizer='adam')
#model.fit(X, y, batch_size=None, epochs=100, steps_per_epoch=1000, validation_steps=200, verbose=1, shuffle=True, validation_split=0.2)

#model = RandomForestRegressor(n_estimators = 1000,                                                                
#                                n_jobs = 2,
#                                verbose = 1,
#                                oob_score = False)
#model = LinearRegression(n_jobs=2)
#model = SymbolicRegressor(population_size=1000,
#                            tournament_size=50,
#                            generations=10, stopping_criteria=0.0,
#                            p_crossover=0.9, p_subtree_mutation=0.00001, p_hoist_mutation=0.00001, p_point_mutation=0.00001,
#                            max_samples=1.0, verbose=1,
#                            #function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),
#                            function_set = ('add', 'sub', 'mul', 'div','sin','cos'),
#                            metric = 'mean absolute error', warm_start=True,
#                            n_jobs = 2, parsimony_coefficient=0.00001, random_state=11)

#model = SymbolicRegressor(n_jobs=2,
#                          generations=20,
#                          verbose=1,
#                          function_set = ('add', 'log', 'max', 'min','div'))
#model = SVR(kernel="rbf", verbose=True)

model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', loss_function='MAE')
n_fold = 5
folds = KFold(n_splits = n_fold, shuffle = True, random_state = 101)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    X_train_f, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train_f, y_valid = y.iloc[train_index], y.iloc[valid_index]    
    model.fit(X_train_f, y_train_f, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=100)

#model.fit(X, y)
#print(model._program)

y_test_p = model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_p)
print("mae:",mae)