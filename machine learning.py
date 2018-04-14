import numpy as np
from sklearn.cross_validation import train_test_split

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
path='features.xlsx'  #give path where extracted features are saved

abc=pd.read_excel(path,header=None)

X=np.array((abc.as_matrix())[1:,1:])
Y=X[:,5]
X=X[:,0:5]
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
y_train = y_train.astype('int')
y_test = y_test.astype('int')
# model=KNeighborsClassifier()

K_value = 3
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
neigh.fit(X_train, y_train)
testt=neigh.predict(X_test)
scoree=accuracy_score(y_test, testt)
print("KNN testing accuracy=",scoree*100)
