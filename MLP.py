#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:44:34 2017

@author: cookie040667
"""

import numpy as np
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3'] 

from sklearn.neural_network import MLPClassifier
X = np.append(trainface,trainnonface,axis=0) #把trainface與trainnonface兩array上下接在一起
X = (np.mean(X)-X)/np.std(X)
y = np.append(np.ones((trainface.shape[0],1)),np.zeros((trainnonface.shape[0],1)),axis=0) #一矩陣放trainface
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)   
y2 = clf.predict(X)
#print(np.sum(y==y2)) 知道有幾個是預測正確的
print(np.sum(y[:,0]==y2)/y.shape[0]) #再除以個數 求得Accuracy *y與y2有一維與二維的問題 y[:,0]作轉換
TX = np.append(testface,testnonface,axis=0)
Ty = np.append(np.ones((testface.shape[0],1)),np.zeros((testnonface.shape[0],1)),axis=0)
Ty2 = clf.predict(TX)
#實驗 改動裡面的設定 找出特徵 x每一排拉出 算他們的mean 標準差 轉成19*19
