import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
style.use('fivethirtyeight')

def predict_example(clf,example_measures):    
    example_measures = example_measures.reshape(len(example_measures),-1)
    prediction = clf.predict(example_measures)
    print(example_measures,prediction)

df = pd.read_csv('yEqX2.txt')

X=np.array(df.drop(['Y'],1))
y=np.array(df['Y'])

print(X)
print(y)

#X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

#clf = neighbors.KNeighborsClassifier(n_jobs=-1)
#clf = svm.SVC(kernel='poly')
#clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='poly')
clf = svm.LinearSVR()
#clf = svm.SVR(kernel='rbf')
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

Xs = [10,11,12,29]
Xss = []
for x in Xs:
    Xss.append([x])

Ys = clf.predict(Xss)

predicted_array = np.array([Xs,Ys])
print(predicted_array)

plt.scatter(df['X'],df['Y'],s=100,c='b')
plt.scatter(Xs,Ys,s=100,c='r')
plt.show()