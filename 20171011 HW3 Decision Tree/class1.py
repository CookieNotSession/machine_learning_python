#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:19:23 2017

@author: cookie040667
"""
import math

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
    num=p1+n1+p2+n2
    num1= p1+n1
    num2= p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)
    #num1/num 代表權重
print(entropy(29,35))
print(InformationGain(21,5,8,30))#A1 tree
print(InformationGain(18,33,11,2))#A2 tree
#結論我們會選擇第一種A1 tree 因為Information Gain比較大 
