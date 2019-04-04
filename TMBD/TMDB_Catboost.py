import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor, Pool

from tqdm import tqdm
import json
import ast

from datetime import datetime

def date(x):
    x=str(x)
    year=x.split('/')[2]
    if int(year)<19:
        return x[:-2]+'20'+year
    else:
        return x[:-2]+'19'+year

def isNaN(x):
    return str(x) == str(1e400 * 0)

def getIsoListFormJson(data, isoKey='id', forceInt=False):
    datas = data.values.flatten()
    ids = []
    for c in (datas):    
        ccc = []
        if isNaN(c) == False:
            c = json.dumps(ast.literal_eval(c))        
            c = json.loads(c)            
            for cc in c:
                if forceInt:
                    ccStr = int(cc[isoKey])
                else:
                    ccStr = str(cc[isoKey])
                ccc.append(ccStr)
        else:
            if forceInt:
                ccc.append(0)
            else:
                ccc.append('0')
        ids.append(ccc)    
    return np.array(ids)

def distributeIdsOverData(data, colName, isoKey='id', forceInt=True):
    arr = getIsoListFormJson(data[colName], isoKey, forceInt)    

    gsi = -1
    for gs in arr:
        gsi += 1
        gs.sort()
        for g in gs:
            gi = gs.index(g)
            try:
                data.loc[gsi, f"{colName}_{gi}"] = float(g)                
            except :
                data.loc[gsi, f"{colName}_{gi}"] = g                
            
    data.drop(colName, axis=1, inplace=True)
    print(f"{colName} distributed over data, cols: {len(data.columns)}")
    
def prepareData(data):        
    data["belongs_to_collection"] = getIsoListFormJson(data["belongs_to_collection"])

    distributeIdsOverData(data,'genres')    
          
    distributeIdsOverData(data,'cast','cast_id')
    
    distributeIdsOverData(data,'crew','name',False)
    
    distributeIdsOverData(data,'Keywords')

    data["Has_HomePage"] = list(map(lambda c: float(c is not np.nan), data["homepage"]))
    data.drop('homepage', axis=1, inplace=True)

    data["IsReleased"] = list(map(lambda c: float(c == "Released"), data["status"]))
    data.drop("status", axis=1, inplace=True)

    data.drop('imdb_id', axis=1, inplace=True)    

    data["original_title_len"] = list(map(lambda c: float(len(str(c))), data["original_title"]))
    data.drop("original_title", axis=1, inplace=True)

    data.drop("overview", axis=1, inplace=True)

    data.drop("poster_path", axis=1, inplace=True)
    
    distributeIdsOverData(data,'production_companies')
    
    distributeIdsOverData(data,'production_countries','iso_3166_1',False)

    data['release_date']=data['release_date'].fillna('1/1/90').apply(lambda x: date(x))
    data['release_date']=data['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
    df_train['release_day']=df_train['release_date'].apply(lambda x:x.weekday())
    df_train['release_month']=df_train['release_date'].apply(lambda x:x.month)
    df_train['release_year']=df_train['release_date'].apply(lambda x:x.year)
    data.drop('release_date', axis=1, inplace=True)
    
    distributeIdsOverData(data,'spoken_languages','iso_639_1',False)

    data["tagline_len"] = list(map(lambda c: float(len(str(c))), data["tagline"]))
    data.drop("tagline", axis=1, inplace=True)

    data["title_len"] = list(map(lambda c: float(len(str(c))), data["title"]))
    data.drop("title", axis=1, inplace=True)    

    data.fillna(0, inplace=True)
    data["budget"] = np.log1p(SimpleImputer(missing_values=0, verbose=1).fit_transform(data["budget"].values.reshape(-1,1)))

    data[LABEL_COL_NAME] = np.log1p(data[LABEL_COL_NAME])

    

TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
SUBMISSON_PATH = "./sample_submission.csv"
LABEL_COL_NAME = "revenue"

if not os.path.exists("all_data.pickle"):
    train = pd.read_csv(TRAIN_DATA_PATH, index_col='id')
    test = pd.read_csv(TEST_DATA_PATH, index_col = 'id')
    all_data = train.append(test)
    print("Preparing All Data")
    prepareData(all_data)    
    all_data.to_pickle("all_data.pickle")
    print("saved all data")
else: 
    all_data = pd.read_pickle("all_data.pickle")
    print("saved all data")

print(all_data.head())
print(all_data.describe())

train = all_data[:len(train)]
test = all_data[len(train):]

print(train.head())
print(train.describe())
print(test.head())
print(test.describe())

X_tr = train.drop(LABEL_COL_NAME, axis = 1)
y_tr = train[LABEL_COL_NAME]

#scaler = StandardScaler()
#scaler.fit(X_tr)
#X_tr = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)

numerical_features = ['budget', 'popularity', 'runtime', 'title_len', 'original_title_len', 'tagline_len']
cat_features = set(X_tr.columns) - set(numerical_features)
cat_features = [list(X_tr.columns).index(c) for c in cat_features]

model_name = 'cat_boost_v2'
model = CatBoostRegressor(iterations=40_000, cat_features=cat_features, verbose=10, early_stopping_rounds=1000)

if not os.path.exists(model_name):
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_tr, y_tr, test_size=0.2, shuffle=True)
    model.fit(X_train, y_train, cat_features, use_best_model=True, eval_set=(X_valid,y_valid))
    model.save_model(model_name)
    print("model saved")
else:
    model.load_model(model_name)
    print("model loaded")

y_test = np.expm1(model.predict(test))
submission = pd.read_csv(SUBMISSON_PATH, index_col='id')
submission[LABEL_COL_NAME] = y_test
submission.to_csv(f'submission_{model_name}.csv')
print(submission.head())
