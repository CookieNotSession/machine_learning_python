#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 19:19:24 2017

@author: cookie040667
"""

#讓程式自己讀取檔案
import os
from PIL import Image #讀圖的函式庫
import numpy as np
import random
import math
os.chdir('CBCL/train/face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
trainface = x.copy()
os.chdir('CBCL/train/non-face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
trainnonface = x.copy()
os.chdir('CBCL/test/face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
testface = x.copy()
os.chdir('CBCL/test/non-face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
testnonface = x.copy()
"""
#lr = learning rate 紀錄學習比例
def BPNN(pf,nf,hn,lr,iteration): #倒傳遞類神經網路
    #test train資料交互訓練 一開始把兩個資料庫接在一起 利用random取完 再重新當作input 這樣就亂掉了(交錯)
    pn = pf.shape[0] #是臉  答案是1的圖片
    nn = nf.shape[0] #不是臉 答案是0的圖片
    fn = pf.shape[1]
    feature = np.append(pf,nf,axis=0) #上下疊在一起 把positive face與 nonface
    target = np.append(np.ones((pn,1)),np.zeros(nn,1),axis=0)
    WI = np.random.normal(0,1,(fn+1,hn)) #361的input 加上1個常數 提供平移
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn) #s裡面為一堆index 亂數取資料 決定training的順序
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1) #input signal
            ho = ins.dot(WI) #hidden node output
            for j in range(hn):
                ho[j] = 1 / (1+math.exp(-ho[j]))
            hs = np.append(ho,1)
            out = hs.dot(WO)
            dk = out*(1-out)*(target[s[i],:]-out)
            dh = np.zeros((hn,1)
            for j in range(hn):  #每一個hidden node
                dh[j] = ho[j]* (1-ho[j]) *WO[j]*dk
                #針對每一個權重都要做調整
            WO = WO + lr * dk * hs
            for j in range(hn):
                WI[:,j] = WI[:,j] + lr * dh[j] * ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO
    return model

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    for i in range(sn):
        ins = np.append(feature[s[i],:],1)
        ho = ins.dot(WI)
        for j in range(hn):
            ho[j] = 1/(1+math.exp(-ho[j]))
        hs = np.append(ho,1)
        out[i] = hs.dot(WO)
    return out
            

network = BPNN(trainface/255,trainnonface/255,20,0.2,1,1)
"""           
