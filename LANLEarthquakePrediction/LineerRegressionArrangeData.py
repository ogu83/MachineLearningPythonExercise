import numpy as np 
import pandas as pd
import os
from tqdm import tqdm

from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

READ_WHOLE_TRAIN_DATA = True
READ_WHOLE_TEST_DATA = True

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"

X_tr = np.array([[]])
y_tr = np.array([[]]);

if os.path.exists(NP_DATA_PATH + "\\X_tr_l.npy") and os.path.exists(NP_DATA_PATH + "\\y_tr_l.npy"):
    X_tr = np.load(NP_DATA_PATH + "\\X_tr_l.npy")
    y_tr = np.load(NP_DATA_PATH + "\\y_tr_l.npy")
else:   
    if READ_WHOLE_TRAIN_DATA:
        chunks = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, chunksize= 10 ** 6)        
        for chunk in tqdm(chunks):
            X = chunk['acoustic_data'];
            y = chunk['time_to_failure'];
            print(X)
            #X_tr_chunk = [[x] for x in X]
            #y_tr_chunk = [[y_] for y_ in y]
            #np.append(X_tr, X_tr_chunk)
            #np.append(y_tr, y_tr_chunk)
    else:
        train = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, nrows=500_000)
        X = train['acoustic_data'];
        y = train['time_to_failure'];
        X_tr = [[x] for x in X]
        y_tr = [[y_] for y_ in y]
        X_tr = np.array(X_tr)
        y_tr = np.array(y_tr)

    np.save(NP_DATA_PATH + "\\X_tr_l.npy", X_tr)
    np.save(NP_DATA_PATH + "\\y_tr_l.npy", y_tr)
    print("numpy arrays saved")

#print(X_tr)
#print(y_tr)

data = X_tr
labels = y_tr
X_train, X_test, y_train, y_test = model_selection.train_test_split(data,labels,test_size=0.2)
rf = RandomForestRegressor(n_estimators=100, #100 trees (Default of 10 is too small)
                          max_features=0.5, #Max number of features each tree can use 
                          min_samples_leaf=30, #Min amount of samples in each leaf
                          random_state=11)
rf.fit(X_train, y_train)
#LOOK FOR ACCURACY AND ERROR
accuracy = rf.score(X_test, y_test)
#PREDICTION AND MAE CHECK
predictions = rf.predict(data[10:])
print("First 10 Labels")
print(labels[10:])
print("Predictions Labels for first 10 data")
print(predictions)
print("Mean Absolute Error")
print(mean_absolute_error(labels, rf.predict(data)))
print("Accuracy")
print(accuracy)