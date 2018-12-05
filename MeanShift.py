import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel("titanic.xls")
original_df = pd.DataFrame.copy(df)

df.drop(['body','name'],1,inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digits_vals = {}
        def convert_to_int(val):
            return text_digits_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digits_vals:
                    text_digits_vals[unique] = x
                    x+=1
            
            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

df.drop(['boat','sex'],1,inplace=True)
print(df.head())

X = np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    pass
    temp_df = original_df[(original_df['cluster_group'] == float(i))]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survival_rate = float(len(survival_cluster))/float(len(temp_df))
    survival_rates[i] = survival_rate

print(survival_rates)
for i in range(len(survival_rates)):
    print("cluster_group:", i, survival_rates[i]*100)
    cluster_i = original_df[ (original_df['cluster_group']==i) ]
    cluster_i_fc = cluster_i[ (cluster_i['pclass']==1) ]
    print(cluster_i_fc.describe())
    #print(cluster_i)
