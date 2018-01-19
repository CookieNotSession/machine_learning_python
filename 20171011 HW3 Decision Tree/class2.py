#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:19:23 2017

@author: cookie040667
"""
import math
import numpy as np

def entropy(p1,n1):
    if(p1==0 and n1==0): #防止p1 n1皆為0 發生0除0問題
        return 1
    value=0
    pp = p1/ (p1+n1) #positive #negative ＃正的機率
    pn = n1/ (p1+n1) #負的機率
    if(pp>0): #判斷後代入Entropy公式
        value -= pp*math.log2(pp)
    if(pn>0):
        value-=pn*math.log2(pn)
    return value

def InformationGain(p1,n1,p2,n2): #IG越大越好
    num=p1+n1+p2+n2 #總數量
    num1= p1+n1 #子樹1的數量
    num2= p2+n2 #子樹2的數量
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)
    #num1/num 代表權重
print(entropy(29,35))
print(InformationGain(21,5,8,30))#A1 tree
print(InformationGain(18,33,11,2))#A2 tree
#結論我們會選擇第一種A1 tree 因為Information Gain比較大 

data = np.loadtxt('PlayTennis.txt',usecols=range(5),dtype=int)
#跑出來的array 前四個是特徵值 第五個是要不要打球
feature = data[:,0:4] #0~3的row
target = data[:,4]-1 #第4row #-1的原因是 跑出來的 1代表打球 2代表不打球 老師想把他歸為 0 1代表

node = dict()
node['data']=range(len(target)) #全部資料都丟到root
Tree = []
Tree.append(node)
i=0
while(i<len(Tree)):
    idx = Tree[i]['data']
    #針對i這個節點找最好的切分點 
    if(sum(target[idx]==0)==0):#Tree[i]裡面的data全部都掉到root(0~13) #把全部加起來 如果都是0 代表都不打
    #把全部sum 代表全部人都不打球
        node[i]['leaf']=1 # leaf=1代表他是葉子
        node[i]['decision']=0 #決策出來結果是 全部都不打
    elif(sum(target[idx])==len(idx)):
        node[i]['leaf']=1 # leaf=1 代表他是葉子
        node[i]['decision']=1 #決策出來的結果是 全部都打
    else: #對所有特徵 for i 就是feature 14*4 所有的特徵都會去try
        bestIG = 0
        for i in range(feature.shape[1]): #i會去跑所有的columns
        #如何檢查是否全部+or全部-
            pool = list(set(feature[idx,i])) #set去找feature中有哪些值 然後在強制轉回list
            for j in range(len(pool)-1): #對每一個特徵i 去切
                thres = (pool[j]+pool[j+1])/2 #兩兩資料的中間 去切
                G1 = []
                G2 = []
                for k in idx:
                    if(feature[k,i]<=thres):
                        G1.append(k)
                    else:
                        G2.append(k)
                thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                if(thisIG>bestIG): #如果這次的IG比上次好 就更新掉
                    bestIG = thisIG
                    bestG1 = G1
                    bestG2 = G2
                    bestthres = thres #切分值
                    bestf = i #最好的特徵值
        if(bestIG>0):
            Tree[i]['leaf']=0
            Tree[i]['selectf']=bestf
            Tree[i]['threshold']=bestthres
            Tree[i]['child']=[len(Tree),len(Tree)+1]
            node = dict()
            node['data']=G1
            Tree.append(node)
            node = dict()
            node['data']=G2
            Tree.append(node)
        else:
            Tree[i]['leaf']=1
            if(sum(target(idx)==1)>sum(target(idx)==0)):
                Tree[i]['decision']=1
            else:
                Tree[i]['decision']=0
    i+=1

