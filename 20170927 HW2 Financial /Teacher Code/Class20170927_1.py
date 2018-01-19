# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
from scipy.stats import norm
import numpy as np #矩陣運算的lib
import matplotlib.pyplot as plt

def blscall(S,L,T,r,sigma):
    d1 = (math.log(S/L)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1- sigma*math.sqrt(T)
    return S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)
    #cumulative distribution function (cdf)
    #從負無限大等於目前的機率

S = 50.0
L = 40.0
T = 2.0
r = 0.08
sigma = 0.2

print(blscall(S,L,T,r,sigma))
print(blscall(S,L,T,r,sigma)+L*math.exp(-r*T)-S)
d1 = (math.log(S/L)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
print(norm.cdf(d1))
print((blscall(S+0.01,L,T,r,sigma)-blscall(S-0.01,L,T,r,sigma))/0.02)

N = 100
dt = T/N

P = np.zeros([10000,N+1]) #模擬一萬次 看機率多少 #P矩陣 一萬的零矩陣
for i in range(10000): 
    P[i,0]=S #每一次的pi i,0從50元開始跑
    for j in range(N): #模擬N次的結果 從這一刻推到下一刻 #從00推到01 從01推到02...10...101..直到結尾
        P[i,j+1]=P[i,j]*math.exp((r-0.5*sigma*sigma)*dt+np.random.normal(0,1,1)*sigma*math.sqrt(dt))
C = 0 #Call Price
drawC = np.zeros([1,100])
for i in range(10000): 
    if(P[i,100]>L): #模擬一百次 最後一筆資料 如果大於執行價格Ｌ(會賺錢的感覺) 放到c中
    #簡而言之 當P價格走的比L還要高時 我們才會去走這份選擇權
        C += (P[i,100]-L)/10000 #如果60 賺到20 因為是期望值 所以除以10000 
    if(i%100==99):
        drawC[(i+1)//100-1]=C/(i+1)*math.exp(-r*T)-blscall(S,L,T,r,sigma)    
print(C*math.exp(-r*T)) #用大量的simulation去模擬股價一萬次 去算最後獲利的期望值 並回推到今天

for i in range(200):
    x=[]
    y=[]
    for j in range(101):
        x.append(j)
        y.append(P[i,j])
        plt.plot(x,y)
plt.grid(True)



        
        
        
        










