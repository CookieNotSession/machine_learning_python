#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:22:51 2017

@author: cookie040667
"""
k=[]
bb=[]
time=int(input())
for g in range(time):
    
    x = input("")
    num = ""
    
    for i in range(len(x)):
        if x[i].isdigit() == False :
            a = x[i]
            printed = False
        else :
            s = ""
            count = 0
            while i < len(x) and x[i].isdigit() == True :
                s += x[i]
                i += 1
            if printed == False:
                k.append(a*int(s))
                printed=True
    bb.append(k)
    k=[]
for index in range(time):
    print("Case "+str(index+1)+": "+('').join(bb[index]))