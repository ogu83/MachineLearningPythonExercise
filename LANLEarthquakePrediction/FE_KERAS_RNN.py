import numpy as np
np.set_printoptions(precision=2)
import pandas as pd
import os

from tqdm import tqdm

from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import keras

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"
MODEL_PATH = f"{DATA_PATH}\\models"

Y_TRAIN_PICKLE = f"{PICKLE_PATH}\\y_train1.pickle"
X_TRAIN_PICKLE = f"{PICKLE_PATH}\\x_train1.pickle"
X_TEST_PICKLE = f"{PICKLE_PATH}\\x_test.pickle"

X_train_scaled = pd.read_pickle(X_TRAIN_PICKLE)
y_tr = pd.read_pickle(Y_TRAIN_PICKLE)
print("Train Data Loaded")
print(X_train_scaled.head())
print(y_tr.head())
print("")

X_test_scaled = pd.read_pickle(X_TEST_PICKLE)
print("X Test Data Loaded")
print(X_test_scaled.head())
print("")

#X_train, X_test, y_train, y_test = model_selection.train_test_split(np.array(X_train_scaled), np.array(y_tr)[:,0], test_size = 0.2)

##KERAS NETWORK MODEL
X = np.array(X_train_scaled)
y = np.array(y_tr)

n_features = len(X_train_scaled.columns)

BATCH_SIZE = 50
EPOCHS = 500
modelName = f"Keras_E{EPOCHS}_6R3_1_ADAM_KERNELN"

model = Sequential(name=modelName)
cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                            histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

if (os.path.exists(f"{MODEL_PATH}\\{model.name}.h5")):    
    model = load_model(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Loaded.", modelName)
else:    
    print("Training Model", modelName)
    model.add(Dense(6, input_dim=n_features, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(6, activation='relu'))      
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    model.fit(X, y, validation_split = 0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[cb], shuffle=True)
    model.save(f"{MODEL_PATH}\\{model.name}.h5")
    print("Model Saved.", modelName)
print(model.summary())

print("Predicting with Model", modelName)
X_pred = np.array(X_test_scaled)
y_pred = model.predict(X_pred)
print("Prediction Completed")

submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id')
submission['time_to_failure'] = y_pred
print(submission.head())
submission.to_csv(f'{DATA_PATH}\\submission{modelName}.csv')
print("Submission File Created")