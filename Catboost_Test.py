import numpy as np
import pandas as pd
import os

from catboost import CatBoostRegressor, Pool,cv
from sklearn.metrics import mean_absolute_error,mean_squared_log_error,mean_squared_error        
from sklearn import preprocessing, model_selection, neighbors, svm


## A catboost category testing, can a descision tree for regression build from a categorical table 

'''
The hidden function reggressor should find
First column is operator (0)Sum (1)Subtract (2)Multiply (3)Divide
Second and third columns are variables
'''
def f(x):
   if (x[0] == 0):
       return x[1]+x[2]
   elif (x[0] >= 1):
       return x[1]-x[2]
   #elif (x[0] == 2):
   #    return x[1]+10*x[2]
   #elif (x[0] == 3):
   #    return 10*x[1]/x[2]

'''
Creates a train data with hidden function
'''
def create_data(data_size=300, max_value = 30):
    arr = np.transpose(np.append(np.random.random_integers(3,size=(1,data_size)),np.random.random_integers(max_value,size=(3,data_size))).reshape(4,data_size))
    for a in arr:
        a[3] = f(a)
    return arr

def train_catboost(data):    
    data = pd.DataFrame(data)
    X_tr = data.drop(3,axis=1)
    y_tr = data[3]    
   
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_tr, y_tr, test_size=0.2, shuffle=True)
    #According to the function the first column of this data is a categoricatl feature others are numerical
    model = CatBoostRegressor(iterations=200, cat_features=[0], verbose=10, early_stopping_rounds=10)
    model.fit(X_train,y_train,eval_set=(X_valid,y_valid))

    #y_pred = model.predict(X_tr)
    #mse = mean_squared_error(y_tr, y_pred)
    #mae = mean_absolute_error(y_tr, y_pred)
    #msle = mean_squared_log_error(y_tr, y_pred)

    #result = pd.DataFrame()
    #result["Predicted"] = y_pred
    #result["Real"] = y_tr
    #result["Error"] = abs((result["Predicted"]-result["Real"])/result["Real"])    
    #print(result)   
    #print(result.describe())

    #print("mae: ", mae)
    #print("mse: ", mse)
    #print("msle: " + msle)     

    return model

def test_catboost(model):
    test = np.array(create_data())
    test = pd.DataFrame(test)

    X_test = test.drop(3,axis=1)
    y_test = test[3]    

    y_test_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)

    result = pd.DataFrame()
    result["Predicted"] = y_test_pred
    result["Real"] = y_test
    result["Error"] = abs((result["Predicted"]-result["Real"])/result["Real"])    
    print(result.head())   
    print(result.tail())   
    #print(result.describe())

    print("mae: ", mae)
    print("mse: ", mse)

def cv_catboost(data):    
    data = pd.DataFrame(data)
    X_tr = data.drop(3,axis=1)
    y_tr = data[3]

    params = {"iterations": 200, "depth": 2, "loss_function": "RMSE", "verbose": True}

    cv_dataset = Pool(data=X_tr, label=y_tr)

    scores = cv(cv_dataset, params, fold_count=5, plot="True")

    return scores

    
train = np.array(create_data())
print("train data", train)
#model = train_catboost(train)
#test_cattest_catboost(model)

scores = cv_catboost(train)
print(scores)