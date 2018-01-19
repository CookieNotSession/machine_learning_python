# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#訊號產生器
import numpy as np
import math
def F1(t):
    return 0.063*(t**3) - 5.284*t*t + 4.887*t + 412 + np.random.normal(0,1)
# F1(t) = 0.063 t3 – 5.284 t2 + 4.887 t + 412 + noise
#觀測所產生出的誤差

def F2(t,A,B,C,D):
    return A*(t**B) + C*math.cos(D*t) + np.random.normal(0,1)

def E(b2,A2,A,B,C,D): #窮舉法: 逼近最好的ABCD Hope this Energy 越小越好 因此min
    sum = 0
    for i in range(1000):
        sum = sum+ abs(F2(A2[i],A,B,C,D) - b2[i]) 
        return sum
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

n = 1000
b2 = np.zeros((n,1))
A2 = np.zeros((n,1))
for i in range(n):
    t = np.random.random()*100
    A2[i] = t #亂數產生的t存進A2中
    b2[i] = F2(t,0.6,1.2,100,0.4)
print(E(b2,A2,0.6,1.2,100,0.4))
print(E(b2,A2,0.6,1.2,99,0.5))




#-5.11~5.11 0.01跳一次 總共有1024個答案 x:D y:energy
#1024*1024的曲面 把他畫出來 (最低點是標準答案)
