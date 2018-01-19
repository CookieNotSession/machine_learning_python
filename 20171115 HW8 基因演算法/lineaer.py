# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#訊號產生器
import numpy as np
def F1(t):
    return 0.063*(t**3) - 5.284*(t**2) + 4.887*t + 412 + np.random.normal(0,1)
# F1(t) = 0.063 t3 – 5.284 t2 + 4.887 t + 412 + noise
#觀測所產生出的誤差

#given t generate one function(t) 已知其為線性回歸
n = 1000  #產生1000個sample
b = np.zeros((n,1)) #產生1*n的0矩陣
A = np.zeros((n,5)) #產生5*n的0矩陣

for i in range(n):
    t = np.random.random()*100 #隨機產生一個t 為0~100之間的數
    b[i] = F1(t) #output signal
    A[i,0] = t**4
    A[i,1] = t**3
    A[i,2] = t**2
    A[i,3] = t
    A[i,4] = 1

x = np.linalg.lstsq(A,b)[0] #最小平方法
print(x)
