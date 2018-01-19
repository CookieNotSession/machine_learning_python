#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:08:16 2017

@author: cookie040667
"""
import numpy as np
import matplotlib.pyplot as plt
import math

def transbeta(beta):
    X0 = 0
    X1 = 1
    Y0 = -1
    Y1 = 1024
    returnbeta = (X1*beta - X1*Y0 -X0*beta + X0*Y1)/(Y1-Y0)
    return returnbeta
def transphi(phi):
    X0 = 0
    X1 = 6.28
    Y0 = -1
    Y1 = 1024
    returnphi = (X1*phi - X1*Y0 -X0*phi + X0*Y1)/(Y1-Y0)
    return returnphi
def transomega(omega):
    X0 = 0
    X1 = 10
    Y0 = -1
    Y1 = 1024
    returnomega = (X1*phi - X1*Y0 -X0*phi + X0*Y1)/(Y1-Y0)
    return returnomega
def transtc(tc):
    X0 = 550
    X1 = 643
    Y0 = -1
    Y1 = 1024
    returnomega = (X1*tc - X1*Y0 -X0*tc + X0*Y1)//(Y1-Y0)
    return returnomega


def F2(tc,t,beta,omega,phi,A,B,C):
    returntc = []
    for i in range(len(t)):
        returntc.append(A+B*(tc-t[i])**beta+C*(tc-t[i])**beta*(math.cos(omega*math.log(tc-t[i])+phi)))
    returntc = np.asarray(returntc)
    return returntc
def E(output,day,A,B,C,beta,omega,tc,phi):
    return np.sum(abs(F2(tc,day,beta,omega,phi,A,B,C)-output))

def getABC(tc,beta,omega,phi):
    b1 = np.zeros((tc,1))
    A1 = np.zeros((tc,3))
    for j in range(tc):
        b1[j] = market[j]
        A1[j,0] = (tc-j)**beta*(math.cos(omega*math.log(tc-j)+phi))
        A1[j,1] = (tc-j)**beta
        A1[j,2] = 1
    X = np.linalg.lstsq(A1,b1)[0]
    C = float(X[0])
    B = float(X[1])
    A = float(X[2])
    return A,B,C


data = np.loadtxt('Bubble.txt',usecols=range(2),dtype=float)
market = []
for i in range(644):
    #market.append(i)
    market.append(data[i][1])
market = np.asarray(market)
market = market.T


people = 1000
g = 1
mutation = 100

output = data[:,1]
output = [math.log(i) for i in output] #取log
output = np.array(output)

pop = np.random.randint(0,2,(people,40))
fit = np.zeros((people,1))
for generation in range(g):
    print(generation)
    for i in range(people):
        gene = pop[i,:]
        beta = np.sum(2**np.array(range(10))*gene[0:10])
        omega = np.sum(2**np.array(range(10))*gene[10:20])
        tc = np.sum(2**np.array(range(10))*gene[20:30])
        phi = np.sum(2**np.array(range(10))*gene[30:40])
        rangebeta = transbeta(beta)
        rangephi = transphi(phi)
        rangeomega = transomega(omega)
        rangetc = transtc(tc)
        A,B,C = getABC(rangetc,rangebeta,rangeomega,rangephi)
        outputlist =[]
        tclist=[]
        for k in range(rangetc):
            outputlist.append(output[k])
        outputlist = np.array(outputlist)
        for k in range(rangetc):
            tclist.append(k)
        tclist = np.array(tclist)
        fit[i]=E(outputlist,tclist,A,B,C,rangebeta,rangeomega,rangetc,rangephi)
    sortf = np.argsort(fit[:,0])
    pop = pop[sortf,:]
    
    for i in range(100,people):
        fid = np.random.randint(0,100)
        mid = np.random.randint(0,100)
        while(mid==fid):
            mid = np.random.randint(0,100)
        mask = np.random.randint(0,2,(1,40))
        son = pop[mid,:]
        father = pop[fid,:]
        son[mask[0,:]==1]=father[mask[0,:]==1]
        pop[i,:] = son
    for i in range(mutation):
        m = np.random.randint(0,people)
        n = np.random.randint(0,40)
        if(pop[m,n]==0):
            pop[m,n]=1
        else:
            pop[m,n]=0

for i in range(people):
    gene = pop[i,:]
    beta = np.sum(2**np.array(range(10))*gene[0:10])
    omega = np.sum(2**np.array(range(10))*gene[10:20])
    tc = np.sum(2**np.array(range(10))*gene[20:30])
    phi = np.sum(2**np.array(range(10))*gene[30:40])
    rangebeta = transbeta(beta)
    rangephi = transphi(phi)
    rangeomega = transomega(omega)
    rangetc = transtc(tc)
    A,B,C = getABC(rangetc,rangebeta,rangeomega,rangephi)
    outputlist =[]
    tclist=[]
    for k in range(rangetc):
        outputlist.append(output[k])
    outputlist = np.array(outputlist)
    for k in range(rangetc):
        tclist.append(k)
    tclist = np.array(tclist)
    fit[i]=E(outputlist,tclist,A,B,C,rangebeta,rangeomega,rangetc,rangephi)
sortf = np.argsort(fit[:,0])
pop = pop[sortf,:]


gene = pop[0,:]
beta = np.sum(2**np.array(range(10))*gene[0:10])
omega = np.sum(2**np.array(range(10))*gene[10:20])
tc = np.sum(2**np.array(range(10))*gene[20:30])
phi = np.sum(2**np.array(range(10))*gene[30:40])
rangebeta = transbeta(beta)
rangephi = transphi(phi)
rangeomega = transomega(omega)
rangetc = transtc(tc)
A,B,C = getABC(rangetc,rangebeta,rangeomega,rangephi)

allday = []
y = []
for j in range(644):
    y.append(market[j])
    allday.append(j)

plt.plot(allday,y,color="g")
littleching=[]
for i in range(rangetc):
    littleching.append(i)
littleching = np.array(littleching)
result = F2(rangetc,littleching,rangebeta,rangeomega,rangephi,A,B,C)
plt.plot(littleching,result)
plt.show()

