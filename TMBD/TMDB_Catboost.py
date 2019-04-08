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
from sklearn.metrics import mean_squared_log_error
from sklearn.impute import SimpleImputer

from catboost import CatBoostRegressor, Pool

from tqdm import tqdm
import json
import ast

from datetime import datetime

TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
SUBMISSON_PATH = "./sample_submission.csv"
LABEL_COL_NAME = "revenue"

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
    for gs in tqdm(arr):
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
    data.drop("overview", axis=1, inplace=True)
    data.drop("poster_path", axis=1, inplace=True)
    data.drop('imdb_id', axis=1, inplace=True)    

    data["belongs_to_collection"] = getIsoListFormJson(data["belongs_to_collection"])

    distributeIdsOverData(data,'genres')
    distributeIdsOverData(data,'cast','cast_id')
    distributeIdsOverData(data,'crew','name',False)
    distributeIdsOverData(data,'Keywords')

    data["Has_HomePage"] = list(map(lambda c: float(c is not np.nan), data["homepage"]))
    data.drop('homepage', axis=1, inplace=True)

    data["IsReleased"] = list(map(lambda c: float(c == "Released"), data["status"]))
    data.drop("status", axis=1, inplace=True)
  
    data["original_title_len"] = list(map(lambda c: float(len(str(c))), data["original_title"]))
    data.drop("original_title", axis=1, inplace=True)
       
    distributeIdsOverData(data,'production_companies')    
    distributeIdsOverData(data,'production_countries','iso_3166_1',False)

    data['release_date']=data['release_date'].fillna('1/1/90').apply(lambda x: date(x))
    data['release_date']=data['release_date'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))
    data['release_day']=data['release_date'].apply(lambda x:x.weekday())
    data['release_month']=data['release_date'].apply(lambda x:x.month)
    data['release_year']=data['release_date'].apply(lambda x:x.year)
    data.drop('release_date', axis=1, inplace=True)
    
    distributeIdsOverData(data,'spoken_languages','iso_639_1',False)

    data["tagline_len"] = list(map(lambda c: float(len(str(c))), data["tagline"]))
    data.drop("tagline", axis=1, inplace=True)

    data["title_len"] = list(map(lambda c: float(len(str(c))), data["title"]))
    data.drop("title", axis=1, inplace=True)    

    data.fillna(0, inplace=True)
    data["budget"] = np.log1p(SimpleImputer(missing_values=0, verbose=1).fit_transform(data["budget"].values.reshape(-1,1)))

    data[LABEL_COL_NAME] = np.log1p(data[LABEL_COL_NAME])



train = pd.read_csv(TRAIN_DATA_PATH, index_col='id')
test = pd.read_csv(TEST_DATA_PATH, index_col = 'id')

if not os.path.exists("all_data.pickle"):   
    ##FILLING MISSIN BUDGET DATA
    train.loc[16,'revenue'] = 192864          # Skinning
    train.loc[90,'budget'] = 30000000         # Sommersby          
    train.loc[118,'budget'] = 60000000        # Wild Hogs
    train.loc[149,'budget'] = 18000000        # Beethoven
    train.loc[313,'revenue'] = 12000000       # The Cookout 
    train.loc[451,'revenue'] = 12000000       # Chasing Liberty
    train.loc[464,'budget'] = 20000000        # Parenthood
    train.loc[470,'budget'] = 13000000        # The Karate Kid, Part II
    train.loc[513,'budget'] = 930000          # From Prada to Nada
    train.loc[797,'budget'] = 8000000         # Welcome to Dongmakgol
    train.loc[819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
    train.loc[850,'budget'] = 90000000        # Modern Times
    train.loc[1112,'budget'] = 7500000        # An Officer and a Gentleman
    train.loc[1131,'budget'] = 4300000        # Smokey and the Bandit   
    train.loc[1359,'budget'] = 10000000       # Stir Crazy 
    train.loc[1542,'budget'] = 1              # All at Once
    train.loc[1542,'budget'] = 15800000       # Crocodile Dundee II
    train.loc[1571,'budget'] = 4000000        # Lady and the Tramp
    train.loc[1714,'budget'] = 46000000       # The Recruit
    train.loc[1721,'budget'] = 17500000       # Cocoon
    train.loc[1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
    train.loc[2268,'budget'] = 17500000       # Madea Goes to Jail budget
    train.loc[2491,'revenue'] = 6800000       # Never Talk to Strangers
    train.loc[2602,'budget'] = 31000000       # Mr. Holland's Opus
    train.loc[2612,'budget'] = 15000000       # Field of Dreams
    train.loc[2696,'budget'] = 10000000       # Nurse 3-D
    train.loc[2801,'budget'] = 10000000       # Fracture

    test.loc[3889,'budget'] = 15000000       # Colossal
    test.loc[6733,'budget'] = 5000000        # The Big Sick
    test.loc[3197,'budget'] = 8000000        # High-Rise
    test.loc[6683,'budget'] = 50000000       # The Pink Panther 2
    test.loc[5704,'budget'] = 4300000        # French Connection II
    test.loc[6109,'budget'] = 281756         # Dogtooth
    test.loc[7242,'budget'] = 10000000       # Addams Family Values
    test.loc[7021,'budget'] = 17540562       #  Two Is a Family
    test.loc[5591,'budget'] = 4000000        # The Orphanage
    test.loc[4282,'budget'] = 20000000       # Big Top Pee-wee

    train.loc[391,'runtime'] = 86 #Il peor natagle de la meva vida
    train.loc[592,'runtime'] = 90 #А поутру они проснулись
    train.loc[925,'runtime'] = 95 #¿Quién mató a Bambi?
    train.loc[978,'runtime'] = 93 #La peggior settimana della mia vita
    train.loc[1256,'runtime'] = 92 #Cipolla Colt
    train.loc[1542,'runtime'] = 93 #Все и сразу
    train.loc[1875,'runtime'] = 86 #Vermist
    train.loc[2151,'runtime'] = 108 #Mechenosets
    train.loc[2499,'runtime'] = 108 #Na Igre 2. Novyy Uroven
    train.loc[2646,'runtime'] = 98 #同桌的妳
    train.loc[2786,'runtime'] = 111 #Revelation
    train.loc[2866,'runtime'] = 96 #Tutto tutto niente niente
    
    test.loc[4074,'runtime'] = 103 #Shikshanachya Aaicha Gho
    test.loc[4222,'runtime'] = 93 #Street Knight
    test.loc[4431,'runtime'] = 100 #Плюс один
    test.loc[5520,'runtime'] = 86 #Glukhar v kino
    test.loc[5845,'runtime'] = 83 #Frau Müller muss weg!
    test.loc[5849,'runtime'] = 140 #Shabd
    test.loc[6210,'runtime'] = 104 #Le dernier souffle
    test.loc[6804,'runtime'] = 145 #Chaahat Ek Nasha..
    test.loc[7321,'runtime'] = 87 #El truco del manco

    all_data = train.append(test)
    print("Preparing All Data")
    prepareData(all_data)    
    all_data.to_pickle("all_data.pickle")
    print("saved all data")
else: 
    all_data = pd.read_pickle("all_data.pickle")
    print("saved all data")

#print(all_data.head())
#print(all_data.describe())

train = all_data[:len(train)]
test = all_data[len(train):]

#print(train.head())
#print(train.describe())
#print(test.head())
#print(test.describe())

X_tr = train.drop(LABEL_COL_NAME, axis = 1)
y_tr = train[LABEL_COL_NAME]

#scaler = StandardScaler()
#scaler.fit(X_tr)
#X_tr = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)

numerical_features = ['budget', 'popularity', 'runtime', 'title_len', 'original_title_len', 'tagline_len']
cat_features = set(X_tr.columns) - set(numerical_features)
cat_features = [list(X_tr.columns).index(c) for c in cat_features]

model_name = 'cat_boost_v3'
model = CatBoostRegressor(iterations=40_000, 
                          learning_rate = 0.01, 
                          depth=10, 
                          l2_leaf_reg = 0.1,
                          cat_features=cat_features,                           
                          verbose=10, early_stopping_rounds=1000)

if not os.path.exists(model_name):
    X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X_tr, y_tr, test_size=0.3, shuffle=True)
    model.fit(X_train, y_train, cat_features, use_best_model=True, eval_set=(X_valid,y_valid))
    model.save_model(model_name)
    print("model saved")
else:
    model.load_model(model_name)
    print("model loaded")

test = test.drop(LABEL_COL_NAME, axis = 1)
y_test = np.expm1(model.predict(test))

submission = pd.read_csv(SUBMISSON_PATH, index_col='id')
submission[LABEL_COL_NAME] = y_test[:-1]
submission.to_csv(f'submission_{model_name}.csv')
print(submission)
