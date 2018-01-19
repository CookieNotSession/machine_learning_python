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
#用4個array將npz的值取回

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]


fn = 0
ftable = [] #建立一張可以找回原圖的table
for y in range(19):
    for x in range(19):
        for h in range(2,20): #2到最大19 最大視窗容許佔滿整個螢幕
                for w in range(2,20):
                    if(y+h<=19 and x+w*2<=19):
                        fn = fn + 1
                        ftable.append([0,y,x,h,w]) #每一個合法特徵 就把yxhw留下 0代表第0類的方塊
for y in range(19):
    for x in range(19):
        for h in range(2,20): #2到最大19 最大視窗容許佔滿整個螢幕
                for w in range(2,20):
                    if(y+h*2<=19 and x+w<=19):
                        fn = fn + 1
                        ftable.append([1,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20): #2到最大19 最大視窗容許佔滿整個螢幕
                for w in range(2,20):
                    if(y+h<=19 and x+w*3<=19):
                        fn = fn + 1
                        ftable.append([2,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20): #2到最大19 最大視窗容許佔滿整個螢幕
                for w in range(2,20):
                    if(y+h*2<=19 and x+w*2<=19):
                        fn = fn + 1
                        ftable.append([3,y,x,h,w])
#sample is N*361 matrix          
#ftable 就是36649張
def fe(sample,ftable,c): #取c個特徵
    ftype = ftable[c][0]
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19))
    if(ftype==0): #白色左邊-右邊 (第一種特徵) 白正的 黑負的
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis = 1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis = 1) #左右
    if(ftype==1): #上下黑白
        output = -np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis = 1) + np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis = 1) #上下
    if(ftype==2): # 白黑白
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis = 1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis = 1)+ np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis = 1)
    if(ftype==3): #白黑黑白
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis = 1) - np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis = 1)- np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis = 1) + np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis = 1)
        
    return output
        #先查表 y一直到 y+h-1 (白色的座標) 跟 x到 x+w(冒號不含 因此不用-1) 這些就是白色的座標 在T的小視窗中 找到
        #flatten就是拉成一維 去把sample所有row的23 21 22的col取出來 往右邊加 就會把2400多人的白色區域一次做完

#白-黑 就會出現一個值 

trpf = np.zeros((trpn,fn))
trnf = np.zeros((trnn,fn))
#tepf = np.zeros((tepn,fn))
#tenf = np.zeros((tenn,fn))

for c in range(fn):
    trpf[:,c] = fe(trainface,ftable,c)
    trnf[:,c] = fe(trainnonface,ftable,c)
    #tepf[:,c] = fe(testface,ftable,c)
    #tenf[:,c] = fe(testnonface,ftable,c)
#heeeeeeeeeeeee
pw = np.ones((trpn,1))/trpn/2 #positive weight 
nw = np.ones((trnn,1))/trnn/2
import math
def WC(pw,nw,pf,nf):
    maxf = max(pf.max(),nf.max()) #在pf中的每一個值
    minf = min(pf.min(),nf.min())
    theta = (maxf - minf)/10+minf #第一刀
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta]) #所有比他還要小的 如果那刀剛好切到 (大於等於) 算同一邊
    #左邊分錯 + 右邊分錯
    polarity = 1
    if(error>0.5):
        error = 1-error
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10): #從第2刀到第9刀
        theta = (maxf - minf)*i/10+minf 
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            error = 1-error
            polarity = 0
        if(error<min_error):
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity
#把t次的結果全部存起來 Strong Classifier
SC = []   
#postive weight
for t in range(10): #取出最好的前10個特徵
    weightsum = np.sum(pw)+np.sum(nw)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0]) #2429個0號特徵 與 4548個0號特徵
    best_feature = 0
    for i in range(1,fn):
        error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i])
        if(error<best_error):
            best_feature = i #取第i個特徵當作最好的
            best_error = error
            best_theta = theta
            best_polarity = polarity
    beta = best_error/(1-best_error)
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha]) #把需要的參數留下
    #把分錯的人 的權重做改變 分對的話 ei=0 權重乘上 beta^(1-0) ; 分錯的人 權重不變
    if(best_polarity==1): #代表右邊正 左邊負
        pw[trpf[:,best_feature]>=best_theta] = pw[trpf[:,best_feature]>=best_theta]*beta #分對的
        nw[trnf[:,best_feature]>=best_theta] = nw[trnf[:,best_feature]>=best_theta]*beta
    else:
        pw[trpf[:,best_feature]<best_theta] = pw[trpf[:,best_feature]<best_theta]*beta #所有小於的正資料 乘以beta
        nw[trnf[:,best_feature]<best_theta] = nw[trnf[:,best_feature]<best_theta]*beta #所有大於的負資料 乘beta
      
trps = np.zeros((trpn,1))
trns = np.zeros((trnn,1))
alpha_sum = 0
for i in range(10):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    if(polarity==1): #代表右邊正 左邊負
        trps[trpf[:,feature]>=theta] = trps[trpf[:,feature]>=theta]+alpha
        trns[trnf[:,feature]>=theta] = trns[trnf[:,feature]>=theta]+alpha
    else:
        trps[trpf[:,feature]<theta] = trps[trpf[:,feature]<theta]+alpha
        trns[trnf[:,feature]<theta] = trns[trnf[:,feature]<theta]+alpha
trps = trps / alpha_sum
trns = trns / alpha_sum
    #1920*1080 跑20個分類器 
    
    
    
    