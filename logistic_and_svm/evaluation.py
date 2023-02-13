import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.svm import SVC


pddata = pd.read_csv(r"D:\临时下载区\machine_learning\machine_learning\bank1.csv", usecols=['"duration"', '"pdays"', 'y'])
pdvalues = pddata.values
X = pdvalues[:,0:2]
Y = pdvalues[:,2]

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3)

logistical_classification = LogisticRegression()
logistical_classification.fit(X_train,y_train)
print(cross_val_score(logistical_classification,X_train,y_train,cv=3,scoring='accuracy'))
print("accuracy:")
print(numpy.mean(cross_val_score(logistical_classification,X_train,y_train,cv=3,scoring='accuracy')))
y_train_pred = cross_val_predict(logistical_classification, X_train, y_train, cv = 3)
print("precision:")
print(precision_score(y_train,y_train_pred))
print("recall:")
print(recall_score(y_train,y_train_pred))
