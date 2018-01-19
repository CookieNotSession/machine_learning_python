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
import matplotlib.pyplot as plt

os.chdir('/Users/cookie040667/Downloads/20171101/CBCL/train/face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
trainface = x.copy()
os.chdir('/Users/cookie040667/Downloads/20171101/CBCL/train/non-face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
trainnonface = x.copy()
os.chdir('/Users/cookie040667/Downloads/20171101/CBCL/test/face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
testface = x.copy()
os.chdir('/Users/cookie040667/Downloads/20171101/CBCL/test/non-face') #滑鼠點到哪個資料夾就讀取
filelist = os.listdir()
x = np.zeros((len(filelist),19*19)) #開一個19*19的開零矩陣
for i in range(len(filelist)):
    IMG = Image.open(filelist[i])
    x[i,:] = np.array(IMG.getdata()) #把每一張照片裡面的0-255的值 (因為是黑白 只有一個19*19) 
testnonface = x.copy()

#homework:選三種learning rate 三種iteration 三種hn個數 去畫training與testng的ROC Curve

def BPNNtrain(pf,nf,hn,lr,iteration):
    pn = pf.shape[0]
    nn = nf.shape[0]
    fn = pf.shape[1]
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0)
    model = dict()
    WI = np.random.normal(0,1,(fn+1,hn))
    WO = np.random.normal(0,1,(hn+1,1))
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1)
            ho = ins.dot(WI)
            for j in range(hn):
                ho[j] = 1/(1+math.exp(-ho[j]))
            hs = np.append(ho,1)
            out = hs.dot(WO)
            out = 1/(1+math.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out)
            dh = ho*(1-ho)*WO[0:hn,0]*dk
            WO[:,0] = WO[:,0] + lr*dk*hs
            for j in range(hn):
                WI[:,j] = WI[:,j] + lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO                                
    return model

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    hn = WI.shape[1]
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1)
        ho = ins.dot(WI)
        for j in range(hn):
            ho[j] = 1/(1+math.exp(-ho[j]))
        hs = np.append(ho,1)
        out[i] = hs.dot(WO)
        out[i] = 1/(1+math.exp(-out[i]))
    return out



def cal(pscore,nscore):
    count1 = 0
    count2 = 0
    yarray=[]
    xarray=[]
    for i in range(1,101,1):
        for j in range(len(pscore)):
            if(pscore[j]>=i/100):
                count1+=1
        count1 = count1/len(pscore)
        yarray.append(count1)
        
        for k in range(len(nscore)):
            if(nscore[k]>=i/100):
                count2+=1
        count2 = count2/len(nscore)
        xarray.append(count2)
        count1 = 0
        count2 = 0
    xarray = np.array(xarray)
    yarray = np.array(yarray)

    return xarray,yarray

def createmodel(trainface,trainnonface,testface,testnonface,varhn,varlr,variteration):
    xarray=[]
    yarray=[]
    network = BPNNtrain(trainface/255,trainnonface/255,varhn,varlr,variteration)
    pscore = BPNNtest(trainface/255,network)
    nscore = BPNNtest(trainnonface/255,network)
    xarray,yarray = cal(pscore,nscore)
    plt.plot(xarray,yarray)
    xarray = []
    yarray = []
    pscore = BPNNtest(testface/255,network)
    nscore = BPNNtest(testnonface/255,network)
    xarray,yarray = cal(pscore,nscore)
    plt.plot(xarray,yarray)
    plt.show()
createmodel(trainface,trainnonface,testface,testnonface,20,0.01,10)
createmodel(trainface,trainnonface,testface,testnonface,20,0.01,20)
createmodel(trainface,trainnonface,testface,testnonface,20,0.01,30)
createmodel(trainface,trainnonface,testface,testnonface,10,0.01,10)
createmodel(trainface,trainnonface,testface,testnonface,20,0.01,10)
createmodel(trainface,trainnonface,testface,testnonface,30,0.01,10)
createmodel(trainface,trainnonface,testface,testnonface,20,0.01,10)
createmodel(trainface,trainnonface,testface,testnonface,20,0.02,10)
createmodel(trainface,trainnonface,testface,testnonface,20,0.03,10)


        
            