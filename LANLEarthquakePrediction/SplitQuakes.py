import pandas as pd
pd.options.mode.use_inf_as_na = True

import numpy as np
from numpy import diff

import os

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec

from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
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

from scipy.stats import moment
from scipy.fftpack import fft, ifft

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
windows = [10,30,50,100,150,300,500,1000,1500,3000,5000,10000,15000,30000,50000]
moments = [1,2,3,4,5,6,7]
percentiles = [0.01,0.09,0.91,0.99]
heavisides = [10**-9, 5*10**-9, 10**-8, 5*10**-8, 10**-7]
heavisides = list(np.append(heavisides, -1*np.array(heavisides)))

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



#model = CatBoostRegressor(iterations = 1000, use_best_model = False, verbose = 10, loss_function= 'MAE', thread_count = 4)

###PRE PROCESSING###
quake_df_all = pd.DataFrame()

if os.path.exists(f'{PICKLE_PATH}\\quake_df_all.pickle'):
    quake_df_all = pd.read_pickle(f'{PICKLE_PATH}\\quake_df_all.pickle')
    print("quake_df_all loaded")    
else:
    quake_index = 0

    for i in tqdm(range(0, len(quakeIndexes) - 2, 1)):
    #for i in range(0, 3, 1):
        quake_index = i
        #print(f"Load Quake {quake_index} - {quakeIndexes[quake_index]}")
        quake_df = pd.read_csv(TRAIN_DATA_PATH, 
                               nrows = quakeIndexes[quake_index+1] - quakeIndexes[quake_index], 
                               skiprows = quakeIndexes[quake_index] + 1, 
                               header = None, names = ["acc","time"], dtype = {"acc":np.int16, "time":np.float16})

        rows = TRAINING_DERIVED_ROW_COUNT
        segments = int(np.floor(quake_df.shape[0] / rows))

        quake_df_tr = pd.DataFrame(index=range(segments), dtype=np.float32)    

        for segment in tqdm(range(segments)):
            seg = quake_df.iloc[segment*rows:segment*rows+rows]
        
            x = pd.Series(seg['acc'].values)
            x_diff = pd.Series(np.append(np.array([0]), np.array(diff(x))))
        
            y = seg['time'].values[-1]
            quake_df_tr.loc[segment, 'time'] = y

            ##Signal Distribution
            quake_df_tr.loc[segment, 'mean'] = x_diff.mean()
            quake_df_tr.loc[segment, 'var'] = x_diff.var()
            quake_df_tr.loc[segment, 'skew'] = x_diff.skew()
            quake_df_tr.loc[segment, 'kurt'] = x_diff.kurt()

            for mom in range(1, 8):            
                quake_df_tr.loc[segment, f'moment_{mom}'] = moment(x_diff, mom)

            ##Precursors
            for p in percentiles:
                quake_df_tr.loc[segment, f'q{int(p*100)}'] = np.quantile(x_diff, p)            

            for h in heavisides:
                hside = np.heaviside(x_diff, h)
                for p in percentiles:
                    quake_df_tr.loc[segment, f'{int(p*100)}_heaviside_{h}'] = np.percentile(hside, p)

            ##Time Corelation Features
            x_diff_fft = pd.Series(fft(x_diff))        
            quake_df_tr.loc[segment, 'fft_var'] = x_diff_fft.var()
            quake_df_tr.loc[segment, 'fft_skew'] = x_diff_fft.skew()
            quake_df_tr.loc[segment, 'fft_kurt'] = x_diff_fft.kurt()

            for w in tqdm(windows):
                x_diff_roll_mean = x_diff.rolling(w).mean().dropna()

                ##Signal Distribution
                quake_df_tr.loc[segment, f'mean_{w}'] = x_diff_roll_mean.mean()
                quake_df_tr.loc[segment, f'var_{w}'] = x_diff_roll_mean.var()
                quake_df_tr.loc[segment, f'skew_{w}'] = x_diff_roll_mean.skew()
                quake_df_tr.loc[segment, f'kurt_{w}'] = x_diff_roll_mean.kurt()

                for mom in range(1, 8):
                    quake_df_tr.loc[segment, f'moment_{mom}_{w}'] = moment(x_diff_roll_mean, mom)

                ##Precursors
                for p in percentiles:
                    quake_df_tr.loc[segment, f'q{int(p*100)}_{w}'] = np.quantile(x_diff_roll_mean, p)            

                for h in heavisides:
                    hside = np.heaviside(x_diff_roll_mean, h)
                    for p in percentiles:
                        quake_df_tr.loc[segment, f'{w}_{int(p*100)}_heaviside_{h}'] = np.percentile(hside, p)
            
                ##Time Corelation Features
                x_diff_roll_fft = pd.Series(fft(x_diff_roll_mean))        
                quake_df_tr.loc[segment, f'{w}_fft_var'] = x_diff_roll_fft.var()
                quake_df_tr.loc[segment, f'{w}_fft_skew'] = x_diff_roll_fft.skew()
                quake_df_tr.loc[segment, f'{w}_fft_kurt'] = x_diff_roll_fft.kurt()

        #quake_df = quake_df[TRAINING_DERIVED_ROW_COUNT:].fillna(0)
        #quake_df = quake_df.groupby(['time']).mean()
        #quake_df['time'] = quake_df.index
        #quake_df = quake_df.reset_index(drop=True)

        #print(quake_df_tr.head())
        #print(quake_df_tr.tail())
        #print(quake_df_tr)
        #print(quake_df_tr.describe())        
       
        quake_df_all = pd.concat(objs=[quake_df_tr, quake_df_all], ignore_index=True)
        del quake_df
        del quake_df_tr
            
    quake_df_all.to_pickle(f'{PICKLE_PATH}\\quake_df_all.pickle')
    print("quake_df_all saved")

print(quake_df_all)
print(quake_df_all.describe())

###TRAIN####

scaler = StandardScaler()

def train():
    tr_X = quake_df_all.drop(["time"], axis=1)
    tr_y = quake_df_all["time"]
    #print(tr_X.head())
    #print(tr_y.head())        
    n_features = len(tr_X.columns)
    
    #modelName = "SplitQuakes_RandomForest_GridSearch_v2"
    #model = RandomForestRegressor(criterion = 'mae')    
    #param_grid = [
    #    #{'n_estimators': [100], 
    #    # 'max_features': [0.05],
    #    # 'min_samples_leaf': [0.005,0.003,0.001],
    #    # 'min_samples_split':[50]},
    #    {'bootstrap': [False], 
    #     'n_estimators': [100], 
    #     'max_features': [0.05,0.03,0.01,0.005],
    #     'min_samples_leaf': [0.001,0.0005,0.0003,0.0001],
    #     'min_samples_split':[50]},
    #]
    #gridSearch = GridSearchCV(model, n_jobs=2, verbose=10, cv=5, scoring='neg_mean_absolute_error', param_grid=param_grid)
    
    modelName = "SplitQuakes_CatBoostRegressor_v3"
    model = CatBoostRegressor(iterations = 30_000, 
                              use_best_model = True, 
                              verbose = 10, 
                              loss_function = 'MAE',                                                             
                              thread_count = 2, 
                              early_stopping_rounds = 3000,
                              random_strength = 200,
                              bagging_temperature = 8,
                              l2_leaf_reg = 8,
                              depth = 10,
                              learning_rate = 0.01)
    
    scaler.fit(tr_X)
    tr_X = pd.DataFrame(scaler.transform(tr_X), columns=tr_X.columns)    

    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(tr_X, tr_y, test_size=0.2, shuffle=True)
    model.fit(X_train, y_train, eval_set=(X_valid,y_valid))
    
    def KERAS_TENSOR():
        BATCH_SIZE = 100
        EPOCHS = 1000
        modelName = f"Keras_Quake_Split_v5"
        model = Sequential(name=modelName)
        cb = keras.callbacks.TensorBoard(log_dir=f'./DNNRegressors/{modelName}/', 
                                    histogram_freq=0, 
                                    batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, 
                                    embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')    

        model.add(Dense(n_features, input_dim=n_features, activation='tanh'))
        for d in range(6):
            model.add(Dense(6,  activation='tanh'))
            model.add(BatchNormalization())
            model.add(Dropout(0.1))    
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae', optimizer='adam')       
    
        if os.path.exists(f'{MODEL_PATH}\\{modelName}.model'):
            model = load_model(f'{MODEL_PATH}\\{modelName}.model')
            print(f'model {modelName} loaded')
        else:
            model.fit(tr_X, tr_y, 
                        batch_size=BATCH_SIZE, 
                        epochs=EPOCHS,                     
                        verbose=1, callbacks=[cb], 
                        validation_split=0.2,
                        shuffle=True)
            model.save(f'{MODEL_PATH}\\{modelName}.model')
            print(f'model {modelName} saved')  

        print(modelName)        
        print(model.summary())    
        
    #KERAS_TENSOR()
    #RANDOM_FOREST()
    
    #model.fit(tr_X, tr_y)
    #gridSearch.fit(tr_X,tr_y)
    #model = gridSearch.best_estimator_

    #print("Grid Search Results")
    #print(gridSearch.cv_results_)

    #print("Grid Search Best Parameters")
    #print(gridSearch.best_params_)

    y_pred = np.array(model.predict(tr_X))
    y_pred = y_pred.reshape(y_pred.shape[0])
    mae = mean_absolute_error(tr_y, y_pred)
    print(f"Model MAE:{mae}")
    result = pd.DataFrame()
    result["Predicted"] = y_pred
    result["Real"] = tr_y
    print(result)

    return model, modelName

model, modelName = train()

###PREDICT TEST DATA###
def predict_test_data():
    submission = pd.read_csv(SUBMISSON_PATH, index_col='seg_id', dtype={"time_to_failure": np.float32})    

    if os.path.exists(f'{PICKLE_PATH}\\quake_df_test_all.pickle'):
        test_df = pd.read_pickle(f'{PICKLE_PATH}\\quake_df_test_all.pickle')
        print("test_df_all loaded")        
    else:
        test_df = pd.DataFrame(index=submission.index, dtype=np.float32)    
        
        for i, seg_id in enumerate(tqdm(submission.index)):  
            seg = pd.read_csv(TEST_DATA_PATH + "\\" + seg_id + '.csv')
                            
            x = pd.Series(seg['acoustic_data'].values)
            x_diff = pd.Series(np.append(np.array([0]), np.array(diff(x))))
        
            acc = pd.Series(seg['acoustic_data'].values)
        
            test_df.loc[seg_id, 'mean'] = x_diff.mean()
            test_df.loc[seg_id, 'var'] = x_diff.var()
            test_df.loc[seg_id, 'skew'] = x_diff.skew()
            test_df.loc[seg_id, 'kurt'] = x_diff.kurt()

            for mom in range(1, 8):            
                test_df.loc[seg_id, f'moment_{mom}'] = moment(x_diff, mom)

            ##Precursors
            for p in percentiles:
                test_df.loc[seg_id, f'q{int(p*100)}'] = np.quantile(x_diff, p)            

            for h in heavisides:
                hside = np.heaviside(x_diff, h)
                for p in percentiles:
                    test_df.loc[seg_id, f'{int(p*100)}_heaviside_{h}'] = np.percentile(hside, p)

            ##Time Corelation Features
            x_diff_fft = pd.Series(fft(x_diff))        
            test_df.loc[seg_id, 'fft_var'] = x_diff_fft.var()
            test_df.loc[seg_id, 'fft_skew'] = x_diff_fft.skew()
            test_df.loc[seg_id, 'fft_kurt'] = x_diff_fft.kurt()

            for w in tqdm(windows):
                x_diff_roll_mean = x_diff.rolling(w).mean().dropna()

                ##Signal Distribution
                test_df.loc[seg_id, f'mean_{w}'] = x_diff_roll_mean.mean()
                test_df.loc[seg_id, f'var_{w}'] = x_diff_roll_mean.var()
                test_df.loc[seg_id, f'skew_{w}'] = x_diff_roll_mean.skew()
                test_df.loc[seg_id, f'kurt_{w}'] = x_diff_roll_mean.kurt()

                for mom in range(1, 8):
                    test_df.loc[seg_id, f'moment_{mom}_{w}'] = moment(x_diff_roll_mean, mom)

                ##Precursors
                for p in percentiles:
                    test_df.loc[seg_id, f'q{int(p*100)}_{w}'] = np.quantile(x_diff_roll_mean, p)            

                for h in heavisides:
                    hside = np.heaviside(x_diff_roll_mean, h)
                    for p in percentiles:
                        test_df.loc[seg_id, f'{w}_{int(p*100)}_heaviside_{h}'] = np.percentile(hside, p)
            
                ##Time Corelation Features
                x_diff_roll_fft = pd.Series(fft(x_diff_roll_mean))        
                test_df.loc[seg_id, f'{w}_fft_var'] = x_diff_roll_fft.var()
                test_df.loc[seg_id, f'{w}_fft_skew'] = x_diff_roll_fft.skew()
                test_df.loc[seg_id, f'{w}_fft_kurt'] = x_diff_roll_fft.kurt()

        test_df.to_pickle(f'{PICKLE_PATH}\\quake_df_test_all.pickle')
        print("test_df_all saved")
     
    print(test_df)
    print(test_df.describe())    

    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    submission.time_to_failure = model.predict(test_df)
    
    print(submission)
    submission.to_csv(f'{DATA_PATH}\\submission{modelName}.csv')
    print("Submission File Created")

predict_test_data()