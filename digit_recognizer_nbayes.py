# -*- coding: utf-8 -*-
"""
Created on Tue May  5 00:23:26 2020

@author: Kunal Jani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#REading the data and storing them into the x and y variables.

train_dataset=pd.read_csv('train.csv')

y=train_dataset.iloc[:,0].values
x=train_dataset.iloc[:,1:].values

#Performing principal component analysis on the dataset

from sklearn.decomposition import PCA
pca=PCA(n_components=155)
datavar=pca.fit_transform(x)

#Normalizing the principal components

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
normalized_datavar = sc.fit_transform(datavar)

#Splitting the data obtained from PCA into training data and testing data

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(normalized_datavar, y, test_size = 1/3, random_state = 0)

#Classifying the test number images using the naive bayes Classifier

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

#Calculate the predicted values as per the parameters calculated by our training model.

y_pred = classifier.predict(x_test)

#Calculation of acuracy by comparing each predicted and actual value.

num_correct=0
for i in range(len(y_pred)):
    if(y_pred[i]==y_test[i]):
        num_correct=num_correct+1 

print(num_correct/len(y_pred))       