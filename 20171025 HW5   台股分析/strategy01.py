#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 00:25:38 2017

@author: cookie040667
"""
#每一分鐘的收 開 高 低
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values #轉二維
tradeday = list(set(TAIEX[:,0]//10000)) #剩下年月日 
tradeday.sort()
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)):
    date = tradeday[i] #先第i天的年月日取出 
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0] #找到屬於那天的每分鐘交易資料的index
    #找到原始資料的第0個column(日期欄位) 
    idx.sort()
    profit[i] = -TAIEX[idx[-1],1] + TAIEX[idx[0],2]
    #idx[-1]:這個交易的最後一筆資料(收盤價)-idx[0]表當天第一筆交易(開盤價)
profit2 = np.cumsum(profit)
print('策略0.1')
print('策略內容: 開盤空一口台指期，收盤時平倉')
plt.plot(profit2)
plt.show()

ans1 = profit2[-1]
ans2 = np.sum(profit>0)/len(profit) #大於0的個數/總個數
ans3 = np.mean(profit[profit>0])
ans4 = np.mean(profit[profit<=0])
plt.hist(profit,bins=100) #切100等分 的直方圖

print('總損益點數:',ans1)
print('勝率:',ans2)
print('賺錢時平均每次獲利點數:',ans3)
print('輸錢時平均每次損失點數:',ans4)
