# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn import datasets
iris = datasets.load_iris()
wicdlist=[]
def kmeans(sample,K,maxiter):
    N = sample.shape[0]
    D = sample.shape[1]
    C = np.zeros((K,D))
    L = np.zeros((N,1))
    L1 = np.zeros((N,1))
    dist = np.zeros((N,K))
    idx = random.sample(range(N),K)
    C = sample[idx,:]
    iter = 0
    while(iter<maxiter):
        for i in range(K):
            dist[:,i] = np.sum((sample-np.tile(C[i,:],(N,1)))**2,1)
        L1 = np.argmin(dist,1)
        if(iter>0 and np.array_equal(L,L1)):
            break
        L = L1
        wicd = np.sum(np.sqrt(np.sum((sample-C[L,:])**2,1)))
    
        for i in range(K):
            idx = np.nonzero(L==i)[0]
            if(len(idx)>0):
                C[i,:] = np.mean(sample[idx,:],0)
        iter += 1
    
    return C,L,wicd




G1 = iris.data
G2 = iris.data
#G1 = (G1-np.tile(G1.mean(0),(G1.shape[0],1)))/np.tile(G1.std(0),(G1.shape[0],1))
wicdlist1=[]
for gg in range(10):

    for i in range(G1.shape[1]):
        meanv = np.mean(G1[:,i])
        stdv = np.std(G1[:,i])
        G1[:,i] = (G1[:,i]-meanv)/stdv
        C,L,wicd1 = kmeans(G1,3,1000)
    wicdlist1.append(wicd1)
    wicd1= np.min(wicdlist1)

wicdlist2=[]
for tt in range(10):
    
    for k in range(G2.shape[1]):
        M=np.max(G2[:,k])
        m=np.min(G2[:,k])
        G2[:,k] = (G2[:,k]-m)/(M-m)
        C,L,wicd2 = kmeans(G2,3,1000)
    wicdlist2.append(wicd2)
    wicd2= np.min(wicdlist2)
    
print(wicd1)
print(wicd2)








