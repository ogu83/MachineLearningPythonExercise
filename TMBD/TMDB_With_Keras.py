import numpy as np
import pandas as pd
import os
import ast

TRAIN_DATA_PATH = "./train.csv"
TEST_DATA_PATH = "./test.csv"
LABEL_COL_NAME = "revenue"

dict_columns = ['belongs_to_collection', 'genres', 'production_companies', 'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def get_json(df):
    global dict_columns
    result=dict()
    for col in dict_columns:
        d=dict()
        rows=df[col].values
        for row in rows:
            if row is None: continue
            for i in row:
                if i['name'] not in d:
                    d[i['name']]=0
                else:
                    d[i['name']]+=1
            result[col]=d
    return result

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

    
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_test = pd.read_csv(TEST_DATA_PATH)
        
dfx = text_to_dict(df_train)
print(dfx)
print(dfx.columns)
print(df_train.columns)

for col in dict_columns:
    df_train[col]=dfx[col]
print(df_train.columns)

df_train['belongs_to_collection'].apply(lambda x:len(x) if x!= {} else 0).value_counts()

#train_dict=get_json(df_train)
#test_dict=get_json(df_train)


