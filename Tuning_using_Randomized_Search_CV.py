#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 14:01:05 2019

@author: amalramakrishnan
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
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
RFReg = RandomForestClassifier(random_state = 1, n_jobs = -1) 
#list of parameters
param_grid = { 
        'n_estimators': [100,500],
        'max_features' : ["auto", "sqrt", "log2"],
        'max_depth' : [x for x in range(10,30)]
}
#search using randomized search cv
CV_rfc = RandomizedSearchCV(estimator=RFReg, param_distributions = param_grid, n_jobs = -1,  cv= 10, n_iter = 50)
CV_rfc.fit(X_train, y_train)
#printing best parameters
print(CV_rfc.best_params_)
