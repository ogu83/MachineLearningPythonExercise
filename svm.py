import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

def predict_example(clf,example_measures):    
    example_measures = example_measures.reshape(len(example_measures),-1)
    prediction = clf.predict(example_measures)
    print(example_measures,prediction)

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True) 

X=np.array(df.drop(['class'],1))
y=np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

#clf = neighbors.KNeighborsClassifier(n_jobs=-1)
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print("accuracy: ", accuracy)
predict_example(clf,np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1],[4,2,1,5,1,2,5,2,1],[1,1,1,1,1,1,1,1,1],[9,9,9,9,9,9,9,9,9]]))
