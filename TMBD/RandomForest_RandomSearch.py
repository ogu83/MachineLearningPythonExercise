import numpy as np
import pandas as pd
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, model_selection, neighbors, svm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

from tqdm import tqdm
import json, ast

def isNaN(x):
    return str(x) == str(1e400*0)

def getIsoListFormJson(data, isoKey):
    datas = data.values.flatten()
    ids = []
    for c in (datas):    
        ccc = []
        if isNaN(c) == False:
            c = json.dumps(ast.literal_eval(c))        
            c = json.loads(c)            
            for cc in c:
                ccStr = str(cc[isoKey])
                ccc.append(ccStr)
        else:
            ccc.append('0')
        ids.append(ccc)
    return ids

def getIdListFromJson(data):
    return getIsoListFormJson(data,'id')

cLB = MultiLabelBinarizer()

def prepareData(data):
    collection_Ids = np.array(getIdListFromJson(data["belongs_to_collection"]))
    cLB.fit(collection_Ids)
    collection_Ids = cLB.transform(collection_Ids)
    cLB_df = pd.DataFrame(collection_Ids, columns=cLB.classes_)
    #data["belongs_to_collection"] = cLB.fit_transform(np.array(getIdListFromJson(data["belongs_to_collection"])))
    data = data.append(cLB_df)


    #data["genres"] = MultiLabelBinarizer().fit_transform(np.array(getIdListFromJson(data["genres"]))).tolist()
    #data["budget"] = SimpleImputer(missing_values=0, verbose=1).fit_transform(data["budget"].values.reshape(-1,1))

    #data["cast"] = getIdListFromJson(data["cast"])
    #data["CastCount"] = list(map(lambda c: float(len(c)), data["cast"]))
    #data["cast"] = MultiLabelBinarizer().fit_transform(np.array(data["cast"])).tolist()

    #data["crew"] = getIdListFromJson(data["crew"])
    #data["CrewCount"] = list(map(lambda c: float(len(c)), data["crew"]))
    #data["crew"] = MultiLabelBinarizer().fit_transform(np.array(data["crew"])).tolist()

    #data["Keywords"] = getIdListFromJson(data["Keywords"])
    #data["Keywords"] = MultiLabelBinarizer().fit_transform(np.array(data["Keywords"]))

    #data["Has_HomePage"] = list(map(lambda c: float(c is not np.nan), data["homepage"]))
    #data.drop('homepage', axis=1, inplace=True)

    #data["IsReleased"] = list(map(lambda c: float(c == "Released"), data["status"]))
    #data.drop("status", axis=1, inplace=True)

    #data.drop('imdb_id', axis=1, inplace=True)

    #data["original_language"] = (LabelBinarizer().fit_transform(data["original_language"])).tolist()

    #data["original_title_len"] = len(data["original_title"])
    #data.drop("original_title", axis=1, inplace=True)

    #data.drop("overview", axis=1, inplace=True)

    #data.drop("poster_path", axis=1, inplace=True)

    #data["production_companies"] = getIdListFromJson(data["production_companies"])
    #data["production_companies"] = MultiLabelBinarizer().fit_transform(np.array(data["production_companies"]))

    #data["production_countries"] = MultiLabelBinarizer().fit_transform(getIsoListFormJson(data["production_countries"],'iso_3166_1')).tolist()

    #data.drop('release_date', axis=1, inplace=True)

    #data["spoken_languages"] = MultiLabelBinarizer().fit_transform(getIsoListFormJson(data["spoken_languages"],'iso_639_1')).tolist()

    #data["tagline_len"] = list(map(lambda c: float(len(str(c))), data["tagline"]))
    #data.drop("tagline", axis=1, inplace=True)

    #data["title_len"] = list(map(lambda c: float(len(str(c))), data["title"]))
    #data.drop("title", axis=1, inplace=True)    


TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
LABEL_COL_NAME = "revenue"

train_orj = pd.read_csv(TRAIN_DATA_PATH, index_col='id')
train = train_orj.copy()

print("Preparing Train Data")
prepareData(train)
print(train.describe())

#X_tr = train.drop("revenue", axis = 1)
#y_tr = train["revenue"]

#model = svm.SVR(verbose=True)
#model.fit(X_tr, y_tr)

