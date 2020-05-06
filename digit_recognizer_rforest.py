# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:33:26 2020

@author: Kunal Jani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_dataset=pd.read_csv('train.csv')

y=train_dataset.iloc[:,0].values
x=train_dataset.iloc[:,1:].values

from sklearn.decomposition import PCA
pca=PCA(n_components=155)
datavar=pca.fit_transform(x)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
normalized_datavar = sc.fit_transform(datavar)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
normalized_datavar = sc.fit_transform(datavar)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(normalized_datavar, y, test_size = 1/3, random_state = 0)

y_train_values=[]
for i in range(10):
    train_values=[]
    for value in y_train:
        if(value==i):
            train_values.append(1)
        else:
            train_values.append(0)
    y_train_values.append(train_values)
y_test_values=[]
for i in range(10):
    test_values=[]
    for value in y_test:
        if(value==i):
            test_values.append(1)
        else:
            test_values.append(0)
    y_test_values.append(test_values)
    
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
y_pred_values=[]
for i in range(10): 
    classifier.fit(x_train, y_train_values[i])
    y_pred_values.append(classifier.predict(x_test))

y_pred=[]
for i in range(len(y_pred_values[0])):
    numbers=[]
    for j in range(10):
        if(y_pred_values[j][i]==1):
            numbers.append(j)
    if(len(numbers)==0):
        numbers.append(-1)
    y_pred.append(numbers)
    
num_correct=0 
for i in range(len(y_test)):
    if(len(y_pred[i])==1 and y_pred[i][0]==y_test[i]):
        num_correct=num_correct+1
        
print(num_correct/len(y_test))