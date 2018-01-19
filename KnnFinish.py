#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:03:17 2017

@author: cookie040667
"""

import numpy as np
import math
import random
from sklearn import datasets
iris = datasets.load_iris()

Data = iris.data
Target = iris.target
CheckBox=[]

def distance(a,b):
        dist = np.sqrt(np.sum((Data[a,:]-Data[b,:])**2,0))
        return dist
ab=np.zeros((150,150))
for a in range(150):
    for b in range(150):
        if a==b :
            ab[a,b] = 1000
        else:
            ab[a,b] = distance(a,b)
            
for a in range(150):
    temp=[]
    for K in range(10):
        temp.append(np.argsort(ab[a,:])[K])
    CheckBox.append(temp)
CheckBox = np.asarray(CheckBox)


for i in range(10):
    Matrix = np.zeros((3,3))
    for a in range(150):
        vote = np.zeros(3,int)
        for knn in range(i+1):
            #Target[CheckBox[a,knn]]
            index = CheckBox[a,knn]
            vote[Target[index]]+=1 
        Matrix[np.argsort(vote)[2],Target[a]]+=1
        #假設今天a=149 vote[0,2,8] 代表花種3對 取出 
        #argsort(vote)[2]取出投票第一多的花種 現在是3 假如原本是第149的資料
        #找原本第149的target應該為多少 (假設為1)
        #那就在 Matrix[3,1]中投一票
        #因為argsort是回傳由小到大的index
        #A=[0,8,2] argsort(A)就會回傳[1,2,0] 裡面的0就是指第0個是最小的
        #因此 argsort(A)的[2]也就是0 就是第三個 就是最大值
        
    print(i+1,'nn:')
    print(Matrix[0,:])
    print(Matrix[1,:])
    print(Matrix[2,:])
    
       
        
            




            

