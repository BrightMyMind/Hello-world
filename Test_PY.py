# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:53:44 2017

@author: 
"""

#%reset

import datetime as dt
from pandas_datareader import data, wb
import os
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import pylab as pl

from sklearn.cross_validation import train_test_split
from pandas.tools.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
import xgboost as xgb

from sklearn import metrics

import time
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout, Flatten, AveragePooling1D, TimeDistributed, Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import regularizers
from keras.layers.embeddings import Embedding
from keras import backend as K
K.set_image_dim_ordering('th')
#from keras.layers.advanced_activations import LeakReLU, PReLU

current_directory = os.getcwd()
os.chdir("C:\Master")


BT1 = pd.read_csv("Test_PY.csv")
BT1 = BT1.drop(['Unnamed: 0'],axis=1)
BT1 = pd.DataFrame(BT1)
#BT1.reset_index()
	
# descriptions
#print(BT1.describe())
BT1 = BT1.sort_values(by='id')
BT1 = BT1.reset_index()
BT1.drop(['index'], axis = 1, inplace = True)
#print(BT1.groupby('Label').size())


############## prepare output data 
BT1_columns = list(BT1.columns.values)
BT1_out = BT1
BT1_out = BT1_out.iloc[:, pd.np.r_[0, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58:68, 70, 82:150, 154, 159, 
                                   164, 169, 174, 179, 184, 189, 194, 199, 204, 209, 214, 219, 224, 229, 
                                   234, 239, 244, 249, 254, 259, 264, 269, 274, 279, 284, 286:289]]
BT1_out_column_names = list(BT1_out.columns.values)
BT1_out.to_csv('Test_PY.csv', mode='w', header=True)

###################################
#BT2 = BT1.iloc[:, pd.np.r_[70, 82:104, 286]]
BT2a = BT1.iloc[:, pd.np.r_[0, 70, 82:114, 286:289]]
BT2a = BT2a.dropna()
column_names = list(BT2a.columns.values)

##################################################

BT2 = BT2a

#BT2 = BT2a[0:6800]
#train, test = BT2[:int(len(BT2) * 0.75)], BT2[int(len(BT2) * 0.75):] 
#train, test = BT2[:int(len(BT2) * 0.99)], BT2[int(len(BT2) * 0.99):] 
#train, test = BT2[:int(len(BT2) * 0.98)], BT2[int(len(BT2) * 0.98 + 40):] 

len_BT  = len(BT2)
#Iteration_start = len_BT - 1500 - 30
Iteration_start = len_BT - 3200 - 30
#Iteration_start = len_BT - 3000 - 30
#train_length = 1000
train_length = 1500

#results_Pred = pd.DataFrame([])
results_Pred = pd.DataFrame()

save_dir="C:\Master"
#batch_size = 32


#i = 0
#for i in range(1900):
for i in range(100):    
    start_train = int(Iteration_start + i)
    end_train = int(Iteration_start + i + train_length)
    
    train, test = BT2[start_train:end_train], BT2[int(end_train+30):int(end_train+30+1)] 
    train = train.loc[(train.Label != 2)]
    
    X_train=train[column_names[0:36]]
    X_test=test[column_names[0:36]]
    Y_train=train[column_names[36]]
    Y_test=test[column_names[36]]

    # Make predictions on validation dataset
#    knn = KNeighborsClassifier()
#    knn.fit(X_train, Y_train)
#    predictions = knn.predict(X_test)
#    PredProb = knn.predict_proba(X_test)
#    print(accuracy_score(Y_test, predictions))
#    print(confusion_matrix(Y_test, predictions))
#    print(classification_report(Y_test, predictions))
    
#    logreg = LogisticRegression()
#    logreg.fit(X_train, Y_train)
#    predictions = logreg.predict(X_test)
#    PredProb = logreg.predict_proba(X_test)

#    classifier = SVC(kernel = 'rbf',probability=True)
#    classifier.fit(X_train, Y_train)
#    predictions = classifier.predict(X_test)
#    PredProb = classifier.predict_proba(X_test)

    #clf = GaussianNB()
    #clf = MultinomialNB()
#    clf.fit(X_train, Y_train)
#    predictions = clf.predict(X_test)
#    PredProb = clf.predict_proba(X_test)

#    xgb_model = xgb.XGBClassifier(max_depth=50, n_estimators=80, learning_rate=0.1, colsample_bytree=.7, gamma=0, reg_alpha=4, objective='binary:logistic', eta=0.3, silent=1, subsample=0.8)
#    xgb_model.fit(X_train, Y_train) 
#    xgb_prediction = xgb_model.predict(X_test)
#    PredProb = xgb_model.predict_proba(X_test)

#    rf = RandomForestClassifier(n_estimators=100)
#    rf.fit(X_train, Y_train)
#    predictions = rf.predict(X_test)
#    PredProb = rf.predict_proba(X_test)

    X_train, Y_train = X_train.values, Y_train.values
    #print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
  
    # reshape input to be 3D [samples, timesteps, features]
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    #print(X_train.shape, Y_train.shape)

    
    model = Sequential()
    #input_shape=X_train.shape[1], X_train.shape[2]
    model.add(LSTM(150, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    
#    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
#
#    model = Sequential()
#    input_shape = X_train.shape[1], X_train.shape[2]
#    model.add(Conv1D(kernel_size=3,filters=18,input_shape=input_shape,activation='relu'))
#    model.add(MaxPooling1D())
#    model.add(Conv1D(kernel_size=3,filters=9,activation='relu'))
#    model.add(MaxPooling1D())
#    model.add(Conv1D(kernel_size=3,filters=3,activation='relu'))
#    model.add(MaxPooling1D())
#    
#    model.add(LSTM(150))
#    model.add(Dense(500, activation='relu'))
#    model.add(Dense(2, activation='softmax'))


#    model = Sequential()
#    model.add(LSTM(100, input_shape=(675, 36), return_sequences=True))
#    model.add(Dropout(0.2))
#    model.add(LSTM(100, return_sequences=True))
#    model.add(LSTM(100, return_sequences=False))
#    model.add(Dropout(0.2))
    
#    input_shape = X_train.shape[1], X_train.shape[2]
#    model.add(Conv1D(kernel_size=3,filters=32,input_shape=input_shape,activation='relu'))
#    model.add(MaxPooling1D())
#    model.add(LSTM(100))
#    model.add(Dropout(0.2))
#    model.add(LSTM(100))
#    model.add(LSTM(100))
#    model.add(Dropout(0.2))
#    model.add(Dense(500, activation='relu'))
    #model.add(Dense(1, activation="linear"))
#    model.add(Dense(2, activation='softmax'))
        
    #opt = optimizers.adam(lr=0.00001)
#    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics=['accuracy'])
    #model.compile(loss="mse", optimizer="adam" )

    opt = optimizers.adam(lr=0.00001)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
    model.summary()    
    print('Model Compiled')

    save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(5)))
    callbacks = [
		ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
	]

    YCat = to_categorical(Y_train)
    col = np.array([1, 3])
    YCat = YCat[:,col]
    
    #model.fit(X_train, YCat, epochs=5, batch_size=128, callbacks=callbacks, verbose=1)
    model.fit(X_train, YCat, epochs=5, batch_size=128, verbose=1)

#    print('Model Training Started')
#    print('Model %s epochs, %s batch size' % (epochs, batch_size))
	
    X_test1, Y_test1 = X_test.values, Y_test.values
    #X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)
    X_test1 = X_test1.reshape((X_test1.shape[0], 1, X_test1.shape[1]))
    
    #X_test1.shape

    #y_score = model.predict(X_test)
    PredProb = model.predict(X_test1)
    predictions = model.predict_classes(X_test1)

#	predicted = model.predict(data)
#	predicted = np.reshape(predicted, (predicted.size,))
   
    Predictions = pd.DataFrame(data=predictions)
    PROB = pd.DataFrame(data=PredProb)
    PROB.columns = ['P_0', 'P_1']
    
    X_test = X_test.reset_index()
    Y_test = Y_test.reset_index()
    
    X_Pred = pd.concat([X_test, Y_test, PROB], axis=1)

    results_Pred = results_Pred.append(X_Pred, ignore_index=True)
    print('Loop %s round' %i)

results_Pred.boxplot(["P_1"],'Label')


BT2_f = BT2.loc[(BT2.Label != 2)]
X_train=BT2_f[column_names[0:36]]
Y_train=BT2_f[column_names[36]]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 10)
rfe = rfe.fit(X_train, Y_train)
print(rfe.support_)
print(rfe.ranking_)

features = pd.DataFrame(column_names[0:36])
support = pd.DataFrame(rfe.support_)
ranking = pd.DataFrame(rfe.ranking_)
rfe1 = pd.concat([features, support, ranking], axis=1)

