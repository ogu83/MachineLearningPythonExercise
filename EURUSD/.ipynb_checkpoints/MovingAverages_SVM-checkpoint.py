import pandas as pd
import numpy as np
import os
import pickle

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection, neighbors, svm

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

import tensorflow as tf
from keras.models import load_model, Sequential
import keras
from keras.layers import Dense, BatchNormalization, Dropout, Activation

import tqdm

from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

EUR_USD_FILE = "EUR_USD Historical Data.csv"
MOVING_AVERAGE_WINDOWS = 100

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

    values = df["Price"].rolling(window=i).values
    df['trend_{i}'] = add_trend_feature(x)    

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
    df[f"KURT{i}_P"] = df["Price"].rolling(window=i).kurt()
    df[f"KURT{i}_O"] = df["Open"].rolling(window=i).kurt()
    df[f"KURT{i}_H"] = df["High"].rolling(window=i).kurt()
    df[f"KURT{i}_L"] = df["Low"].rolling(window=i).kurt()

#df = df[(MOVING_AVERAGE_WINDOWS-2):]
df = df[:-1]
df.fillna(value=0, axis=1, inplace=True)

print(df.head())
print(df.tail())
print(df.describe())


##y_tr = df["Buy"]
##X_tr = df.drop(["PriceF1", "Buy", "Date"], axis=1)

##Preprocessing Data
#X = np.array(df.drop(["Price", "High", "Low", "Date"], 1).astype(float))
#X = preprocessing.scale(X)
#y = np.array(df['Price'])
##X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#n_features = (X.shape[1])
#print("n_features:",n_features,",X shape:",X.shape)

#def ExecKerasModel():
#    BATCH_SIZE = 1000
#    EPOCHS = 1000
#    #modelName = f"Keras_v5"
#    #MaxBalance $314418 at Nov 12, 2008
#    #Daily Income: $18
#    #Monthly Income: $526
#    #Gross Loss: $8224407, Gain: $8345896

#    modelName = f"Keras_v7"
#    model = Sequential(name=modelName)
#    cb = keras.callbacks.TensorBoard(log_dir=f'./Keras/{modelName}/', 
#                                histogram_freq=0, 
#                                batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
#                                embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

#    if (os.path.exists(f"{model.name}.h5")):    
#        model = load_model(f"{model.name}.h5")
#        print("Model Loaded.", modelName)
#    else:    
#        print("Training Model", modelName)
#        model.add(Dense(n_features, input_dim=n_features, activation='relu'))    
    
#        model.add(Dense(40))
#        model.add(BatchNormalization())
#        model.add(Activation('tanh'))
#        model.add(Dropout(0.25))  

#        model.add(Dense(10))
#        model.add(BatchNormalization())
#        model.add(Activation('tanh'))
#        model.add(Dropout(0.25))  
    
#        model.add(Dense(1, activation='linear'))    
#        model.compile(loss=['mse'], optimizer='adam', metrics=['mse', 'mae'])
#        model.fit(X, y, validation_split = 0.3, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)
#        model.save(f"{model.name}.h5")
#        print("Model Saved.", modelName)
#    print(model.summary())
#    return model

#def ExecRandomForestRegressor():
#    modelName = f"RandomForest_1000Estimator"

#    if (os.path.exists(f"{modelName}.pickle")):    
#        pickle_in = open(f'{modelName}.pickle','rb')
#        model = pickle.load(pickle_in)
#        print("Model Loaded")
#    else:
#        model = RandomForestRegressor(n_estimators=1000,n_jobs=2,random_state=11,verbose=1)
#        model.fit(X,y)
#        with open(f"{modelName}.pickle",'wb') as f:
#            pickle.dump(model, f)
#            print('Model Saved')        

#    return model

#def ExecCatBoost():   
#    modelName = f"CatBoost2000"

#    if (os.path.exists(f"{modelName}.pickle")):    
#        pickle_in = open(f'{modelName}.pickle','rb')
#        model = pickle.load(pickle_in)
#        print("Model Loaded")
#    else:
#        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#        model = CatBoostRegressor(iterations=2000)
#        model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=[], use_best_model=True, verbose=True)
#        with open(f"{modelName}.pickle",'wb') as f:
#            pickle.dump(model, f)
#            print('Model Saved')    
       
#    return model
    
##model = ExecKerasModel()
##model = ExecRandomForestRegressor()
#model = ExecCatBoost()

##STRATEGY TESTER
#print("STRATEGY TESTER")
#y_pred = model.predict(X)
#Init_Balance = 10_000
#Balance = Init_Balance
#Margin_Call = 0.1
#Leverange = 5
#MaxBalance = 0
#MaxBalanceDate = None
#StartDate = None
#EndDate = None
#Days = 0
#GrossLoss = 0
#GrossGain = 0
#BalanceGraph = []

##for index, row in df.iterrows():
#for index, row in df[-365*1:].iterrows():
##for index, row in df[-365*2:][:365*1].iterrows():
#    buy = y_pred[index-1] > row["Open"]
#    diff = (row["Price"] - row["Open"]) 
#    change = diff / row["Price"]
#    lossGain = change * Balance
#    if buy == False:
#        lossGain *= -1
#    lossGain *= Leverange
#    Balance += lossGain
#    BalanceGraph.append(Balance)

#    if (lossGain>0):
#        GrossGain += abs(lossGain)
#    else:
#        GrossLoss += abs(lossGain)

#    if (diff != 0):
#        Days += 1
#        if (StartDate == None):
#            StartDate = row["Date"]
#        if buy: 
#            actionStr = "Buy" 
#        else: 
#            actionStr = "Sell"
#        print(row["Date"], f"Open: {row['Open']}", f"Close: {row['Price']}", f"{actionStr}", f"Gain: ${round(lossGain)}", f"Balance: ${round(Balance)}")

#    if (MaxBalance < Balance):
#        MaxBalance = Balance
#        MaxBalanceDate = row["Date"]

#    EndDate = row["Date"]

#    if (Balance < Margin_Call*Init_Balance):        
#        break       

#print(f"MaxBalance ${round(MaxBalance)} at {MaxBalanceDate}")

#DailyIncome = (Balance - Init_Balance) / (Days)
#print(f"Daily Income: ${round(DailyIncome)}")
#MontlyIncome = DailyIncome * 30
#print(f"Monthly Income: ${round(MontlyIncome)}")
#print(f"Gross Loss: ${round(GrossLoss)}, Gain: ${round(GrossGain)}")

#plt.plot(BalanceGraph)
#plt.show()