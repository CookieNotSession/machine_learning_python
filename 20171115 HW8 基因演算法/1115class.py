#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 18:33:54 2017

@author: dodokey
"""
import numpy as np
import math

#Liner Regression
#訊號產生器
def F1(t):#input sigmal
    return 0.063*(t**3)-5.248*(t**2)+4.887*t+412+np.random.normal(0,1)

#F1(t) = 0.063 t3 – 5.284 t2 + 4.887 t + 412 + noise
#假設 我知道她最高是四次方，且t是整數
n=10000
b=np.zeros((n,1))
A=np.zeros((n,5))


for i in range(n):
    #產生output signal
    t=np.random.random()*100
    b[i]=F1(t)
    #intput signal
    A[i,0]=t**4
    A[i,1]=t**3
    A[i,2]=t**2
    A[i,3]=t
    A[i,4]=1
    
X=np.linalg.lstsq(A,b)[0]
print(X)



#訊號產生器
#F2(t) = 0.6 t1.2 + 100 cos(0.4t) + noise;


def F2(t,A,B,C,D):
    return A*(t**B)+C*math.cos(D*t)+np.random.normal(0,1)

#窮舉法
def E(b2,A2,A,B,C,D):
    sum=0
    for i in range(1000):
        sum=sum+abs(F2(A2[i],A,B,C,D)-b2[i])
    return sum
    
    
n=1000
b2=np.zeros((n,1))
A2=np.zeros((n,1))
for i in range(n):
    t=np.random.random()*100
    A2[i]=t
    b2[i]=F2(t,0.6,1.2,100,0.4)

print(E(b2,A2,0.6,1.2,100,0.4))
print(E(b2,A2,0.6,1.2,100,0.5))
