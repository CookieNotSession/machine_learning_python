# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#每一分鐘的時間0 收1 開2 高3 低4
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values #轉二維
tradeday = list(set(TAIEX[:,0]//10000)) #剩下年月日 
tradeday.sort()
profit = np.zeros((len(tradeday),1))
for i in range(len(tradeday)): #把分鐘資料找出來
    date = tradeday[i] #取最後一筆收盤價 與 第一筆開盤價 
    idx = np.nonzero(TAIEX[:,0]//10000==date)[0]
    idx.sort()
    p1 = TAIEX[idx[0],2] #開盤第一分鐘的開盤價 我買的價格
    idx2 = np.nonzero(TAIEX[idx,4]<=p1-30)[0] #把那天每一分鐘的最低價拿出來 看有沒有人小於等於p1-30
    #就把他的index拿出來 而這個index不是原始的index 而是這300分鐘的index
    #idx3 = np.nonzero(TAIEX[idx,3]>=p1-30)[0]
    if(len(idx2)==0): #如果idx2是空集合 代表當天資料都沒觸及到這個低點
        p2 = TAIEX[idx[-1],1] #因此回傳當天最後一筆收盤價 
    else: 
        p2 = TAIEX[idx[idx2[0]],1] #否則就回傳踩中低點 那一刻 的收盤價 (趕快退出！)
    #去找第一筆碰到30這個低點的資料 如果有就把碰到的這筆資料的收盤價賣出
    #如果一直都沒有資料碰到30低點 那就用當天的收盤價賣出 (收盤平倉)
    profit[i] = p1-p2
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