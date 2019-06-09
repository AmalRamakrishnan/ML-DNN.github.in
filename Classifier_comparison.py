#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:48:53 2019

@author: amalramakrishnan
"""

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier

# load pima indians diabetes dataset 
filename = 'pima-indians-diabetes.data.csv'
dataframe = read_csv(filename)
array = dataframe.values
# split into input (X) and output (Y) variables
X = array[:,0:8]
y = array[:,8]
#Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX = scaler.fit_transform(X)
#Classifiers(LR, LDA, KNN, CART, NB, RF)
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=500, max_depth=27, max_features='sqrt', random_state=0)))
#evaluate each model in turn 
aresults = []
fresults = []
presults = []
rresults = []
aresults = []
names = []
#Iterate Classifiers over Dataset using KFold and Train Test Split
for name, model in models:
    kfold = KFold(n_splits=10, random_state=0,shuffle=True)
    cv_accuracy = cross_val_score(model, rescaledX, y, cv=kfold, scoring='accuracy')
    cv_f1 = cross_val_score(model, rescaledX, y, cv=kfold, scoring='f1')
    cv_precision = cross_val_score(model, rescaledX, y, cv=kfold, scoring='precision')
    cv_recall = cross_val_score(model, rescaledX, y, cv=kfold, scoring='recall')
    aresults.append(cv_accuracy)
    fresults.append(cv_f1)
    presults.append(cv_precision)
    rresults.append(cv_recall)
    names.append(name)
    accuracy = (cv_accuracy.mean())
    f1 =(cv_f1.mean())
    precision = (cv_precision.mean())
    recall = (cv_recall.mean())
    print('\nKFold-',name)
    print('Accuracy   :', accuracy)
    print('F1         :', f1)
    print('Precision  :', precision)
    print('Recall     :', recall)
    X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.4, random_state=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    recall=((matrix[1,1])/(matrix[1,1]+matrix[1,0]))
    Accuracy=((matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[0,1]+matrix[1,0]+matrix[1,1]))
    Precision=((matrix[1,1])/(matrix[1,1]+matrix[0,1]))
    F_measure=((2*Precision*recall)/(Precision+recall))
    print('\nTrain-Test-Split-',name)
    #print(name)
    print('Accuracy   :',Accuracy)
    print('F1         :',F_measure)
    print('Precision  :',Precision)
    print('Recall     :',recall)
    #print('FPR        :',fpr)