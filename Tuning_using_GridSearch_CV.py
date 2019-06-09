#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:01:05 2019

@author: amalramakrishnan
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# load pima indians diabetes dataset 
filename = 'pima-indians-diabetes.data.csv'
dataframe = read_csv(filename)
array = dataframe.values
X = array[:,0:8]
y = array[:,8]
#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)

test_size = 0.33
seed = 4
X_train,X_test,y_train,y_test = train_test_split(rescaledX,y,test_size = test_size, random_state = seed)
#RandomForest
rfc=RandomForestClassifier(random_state=4)
#list of parameters
param_grid = { 
    'n_estimators': [100, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7],
    'criterion' :['gini', 'entropy']
}
#search using randomized search cv
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
#print best parameters
print(CV_rfc.best_params_)