# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#每一分鐘的 時間0 收1 開2 高3 低4
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values
tradeday = list(set(TAIEX[:,0]//10000)) #剩下年月日 
tradeday.sort()
profit = []
for i in range(len(tradeday)): #把分鐘資料找出來
    date = tradeday[i] #取最後一筆收盤價 與 第一筆開盤價 
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] #開盤第一分鐘的開盤價 我買的價格
    idx2 = np.nonzero(TAIEX[idx,3]>=p1+30)[0] #符合買進一口
    idx3 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #符合放空一口
    if(len(idx2)==0 and len(idx3)==0): #沒遇到30與-30 不出手
        continue #不買
    elif(len(idx3)==0):#買進
        p2 = TAIEX[idx[idx2[0]],1] #買
        p3 = TAIEX[idx[-1],1] #賣
        profit.append(p3-p2)
        
    elif(len(idx2)==0): #放空
        p3 = TAIEX[idx[-1],1] #賣
        p2 = TAIEX[idx[idx3[0]],1] #買
        profit.append(p2-p3)
    elif(idx2[0]<idx3[0]):
        p2 = TAIEX[idx[idx2[0]],1] #買
        p3 = TAIEX[idx[-1],1] #賣
        profit.append(p3-p2)
    else:
        p3 = TAIEX[idx[-1],1] #賣
        p2 = TAIEX[idx[idx3[0]],1] #買
        profit.append(p2-p3)
profit = np.array(profit)
profit2 = np.cumsum(profit)
plt.plot(profit2)
plt.show()
#不賠不賺都算虧錢 
ans1 = profit2[-1]
ans2 = np.sum(profit>0)/len(profit) #大於0的個數/總個數
ans3 = np.mean(profit[profit>0])
ans4 = np.mean(profit[profit<=0])
plt.hist(profit,bins=100) #切100等分 的直方圖
print('總損益點數:',ans1)
print('勝率:',ans2)
print('賺錢時平均每次獲利點數:',ans3)
print('輸錢時平均每次損失點數:',ans4)

