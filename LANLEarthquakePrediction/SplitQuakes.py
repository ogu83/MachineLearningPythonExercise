import pandas as pd
pd.options.mode.use_inf_as_na = True
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
import keras

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

DATA_PATH = "D:\\LANLEarthquakeData"
TRAIN_DATA_PATH = f"{DATA_PATH}\\train.csv"
TEST_DATA_PATH = f"{DATA_PATH}\\test"
SUBMISSON_PATH = f"{DATA_PATH}\\sample_submission.csv"
TRAINING_DERIVED_ROW_COUNT = 150_000
READ_WHOLE_TRAIN_DATA = True
READ_WHOLE_TEST_DATA = True
NP_DATA_PATH = f"{DATA_PATH}\\np"
PICKLE_PATH = f"{DATA_PATH}\\pickle"
MODEL_PATH = f"{DATA_PATH}\\models"


quakeIndexes = [0,5656574,50085878,104677356,138772453,187641820,218652630,245829585,307838917,338276287,375377848,419368880,461811623,495800225,528777115,585568144,621985673]

windows = []

#chunks = pd.read_csv(TRAIN_DATA_PATH, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}, chunksize= 10 ** 6)
#for chunk in tqdm(chunks):
#    df = chunk
#    df["time_diff"] = df["time_to_failure"].shift(1) - df["time_to_failure"]
#    split_index.append(df[df["time_diff"]<0].index.values)

#print(split_index)

###MODEL DEFINITION###

#model=MLPRegressor(activation='tanh',
#                   alpha=0.1,
#                   learning_rate='adaptive',
#                   shuffle=True,
#                   verbose=True,
#                   warm_start=True,
#                   early_stopping =True)

#model = NuSVR(gamma='scale', verbose=True)
#model = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01, verbose=True)

#model = RandomForestRegressor(n_estimators = 0,
#                                criterion = 'mae',   
#                                max_features= 0.5,
#                                min_samples_leaf = 30,
#                                n_jobs = 1,
#                                verbose = 100,
#                                oob_score = False,
#                                warm_start = True)


BATCH_SIZE = TRAINING_DERIVED_ROW_COUNT//10000
EPOCHS = 100
modelName = f"Keras_Quake_Split_v1"
model = Sequential(name=modelName)
cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                            histogram_freq=0, 
                            batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                            embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

n_features = 9+9*len(windows)

model.add(Dense(n_features*2, input_dim=n_features, activation='tanh'))
for d in range(3):
    model.add(Dense(n_features, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))    
model.add(Dense(1, activation='linear'))
model.compile(loss='mae', optimizer='adam')



#def tanh(x):
#    return np.tanh(x);
#def sinh(x):
#    return np.sinh(x);
#def cosh(x):
#    return np.cosh(x);

#gp_tanh = make_function(tanh,"tanh",1)
#gp_sinh = make_function(sinh,"sinh",1)
#gp_cosh = make_function(cosh,"cosh",1)

#model = SymbolicRegressor(population_size=500,
#                            tournament_size=20,
#                            generations=0, stopping_criteria=1.9,
#                            p_crossover=0.9, p_subtree_mutation=0.01, p_hoist_mutation=0.01, p_point_mutation=0.01,
#                            max_samples=0.1, verbose=1,
#                            #function_set = ('add', 'sub', 'mul', 'div', gp_tanh, 'sqrt', 'log', 'abs', 'neg', 'inv','max', 'min', 'tan', 'cos', 'sin'),
#                            function_set = (gp_tanh, 'add', 'sub', 'mul', 'div'),
#                            metric = 'mean absolute error', warm_start=True,
#                            n_jobs = 1, parsimony_coefficient=0.001, random_state=11)



#model = CatBoostRegressor(iterations = 1000, use_best_model = False, verbose = 10, loss_function= 'MAE', thread_count = 1)


###TRAINING QUAKES###
quake_index = 0
for i in range(0, len(quakeIndexes) - 2, 1):
#for i in range(0, 1, 1):
    quake_index = i
    print(f"Load Quake {quake_index} - {quakeIndexes[quake_index]}")
    quake_df = pd.read_csv(TRAIN_DATA_PATH, 
                           nrows = quakeIndexes[quake_index+1] - quakeIndexes[quake_index], 
                           skiprows = quakeIndexes[quake_index] + 1, 
                           header = None, names = ["acc","time"], dtype = {"acc":np.int16, "time":np.float16})

    quake_df["mean"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).mean()
    quake_df["sum"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).sum()
    quake_df["median"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).median()
    quake_df["var"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).var()
    quake_df["std"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).std()
    quake_df["min"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).min()
    quake_df["max"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).max()    
    quake_df["skew"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).skew()
    quake_df["kurt"] = quake_df["acc"].rolling(window=TRAINING_DERIVED_ROW_COUNT).kurt()

    for w in tqdm(windows):
        quake_df[f"{w}_mean"] = quake_df["acc"].rolling(window=w).mean()
        quake_df[f"{w}_sum"] = quake_df["acc"].rolling(window=w).sum()
        quake_df[f"{w}_median"] = quake_df["acc"].rolling(window=w).median()
        quake_df[f"{w}_var"] = quake_df["acc"].rolling(window=w).var()
        quake_df[f"{w}_std"] = quake_df["acc"].rolling(window=w).std()
        quake_df[f"{w}_min"] = quake_df["acc"].rolling(window=w).min()
        quake_df[f"{w}_max"] = quake_df["acc"].rolling(window=w).max()    
        quake_df[f"{w}_skew"] = quake_df["acc"].rolling(window=w).skew()
        quake_df[f"{w}_kurt"] = quake_df["acc"].rolling(window=w).kurt()

    quake_df = quake_df[TRAINING_DERIVED_ROW_COUNT:].fillna(0)
    quake_df = quake_df.groupby(['time']).mean()
    quake_df['time'] = quake_df.index
    quake_df = quake_df.reset_index(drop=True)

    print(quake_df.head(1))
    print(quake_df.tail(1))
    #print(quake_df)
    print(quake_df.describe())

    #fig = plt.figure(constrained_layout=True)
    #gs = gridspec.GridSpec(3, 3, figure=fig)
    #ax00 = fig.add_subplot(gs[0, 0])
    #ax00.plot(quake_df["mean"])
    #ax00.set_ylabel('mean')
    #ax01 = fig.add_subplot(gs[0, 1])
    #ax01.plot(quake_df["sum"])
    #ax01.set_ylabel('sum')
    #ax02 = fig.add_subplot(gs[0, 2])
    #ax02.plot(quake_df["median"])
    #ax02.set_ylabel('median')
    #ax10 = fig.add_subplot(gs[1, 0])
    #ax10.plot(quake_df["var"])
    #ax10.set_ylabel('var')
    #ax11 = fig.add_subplot(gs[1, 1])
    #ax11.plot(quake_df["std"])
    #ax11.set_ylabel('std')
    #ax12 = fig.add_subplot(gs[1, 2])
    #ax12.plot(quake_df["min"])
    #ax12.set_ylabel('min')
    #ax20 = fig.add_subplot(gs[2, 0])
    #ax20.plot(quake_df["max"])
    #ax20.set_ylabel('max')
    #ax21 = fig.add_subplot(gs[2, 1])
    #ax21.plot(quake_df["skew"])
    #ax21.set_ylabel('skew')
    #ax22 = fig.add_subplot(gs[2, 2])
    #ax22.plot(quake_df["kurt"])
    #ax22.set_ylabel('kurt')
    #plt.show()

    #plt.plot(quake_df["time"])
    #plt.show()    

    def train():
        print("----------------------------------------------------------")
        print(f"Train Quake {quake_index} - {quakeIndexes[quake_index]}")
        tr_X = quake_df.drop(["acc","time"], axis=1)
        tr_y = quake_df["time"]
        print(tr_X.head())
        print(tr_y.head())

        #model.n_estimators=(quake_index+1) * 10
        #model.fit(tr_X, tr_y)

        #model.train_on_batch(tr_X, ty_y)
        model.fit(tr_X, tr_y, 
                  batch_size=BATCH_SIZE, 
                  epochs=(quake_index+1)*EPOCHS, 
                  initial_epoch=quake_index*EPOCHS,
                  verbose=1, callbacks=[cb], 
                  validation_split=0.1,
                  shuffle=True)
        model.save(f'{MODEL_PATH}\\{modelName}.model')

        #model.generations += 10
        #model.fit(tr_X,tr_y)
        #print(model._program)

        #model.fit(tr_X, tr_y)

        #y_pred = model.predict(tr_X)
        #mae = mean_absolute_error(tr_y, y_pred)
        #print(f"Quake:{quake_index} Model MAE:{mae}")
        #result = pd.DataFrame()
        #result["Predicted"] = y_pred
        #result["Real"] = tr_y
        #print(y_pred)

    train()
    del quake_df

###PREDICT TEST DATA###
def predict_test_data():
    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id', dtype={"time_to_failure": np.float32})
    for i, seg_id in enumerate(tqdm(submission.index)):  
        seg = pd.read_csv(TEST_DATA_PATH + "\\" + seg_id + '.csv')
        
        acc = pd.Series(seg['acoustic_data'].values)
        test_df = pd.DataFrame()
        test_df["acc"] = acc[-1:]    
        test_df["mean"] = acc.mean()
        test_df["sum"] = acc.sum()
        test_df["median"] = acc.median()
        test_df["var"] = acc.var()
        test_df["std"] = acc.std()
        test_df["min"] = acc.min()
        test_df["max"] = acc.max()
        #test_df["cov"] = acc.cov()
        test_df["skew"] = acc.skew()
        test_df["kurt"] = acc.kurt()
        test_df.drop("acc",axis=1,inplace=True)
     
        #print(test_df)
        time = model.predict(test_df)   
        submission.time_to_failure[i] = time
    
    print(submission)
    submission.to_csv(f'{DATA_PATH}\\submission{modelName}.csv')
    print("Submission File Created")

predict_test_data()


