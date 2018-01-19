# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:02:04 2017

@author: USER
"""
import math
import matplotlib.pyplot as plt

def blscall(S,L,T,r,sigma):
    d1 = (math.log(S/L)+(r+0.5*sigma*sigma)*T)/(sigma*math.sqrt(T))
    d2 = d1- sigma*math.sqrt(T)
    return S*norm.cdf(d1)-L*math.exp(-r*T)*norm.cdf(d2)

def Bisection(left,right,S,L,T,r,call,error):
    center = (left+right)/2
    if((right-left)/2<error):
        return center
    if((blscall(S,L,T,r,left)-call)*(blscall(S,L,T,r,center)-call)<0):
        return Bisection(left,center,S,L,T,r,call,error)
    else:
        return Bisection(center,right,S,L,T,r,call,error)

Q3Bi=[]
def Bisection2(left,right,S,L,T,r,call,iteration):
    center = (left+right)/2
    Q3Bi.append(center)
    if(iteration==0):
        return center
    if((blscall(S,L,T,r,left)-call)*(blscall(S,L,T,r,center)-call)<0):
        return Bisection2(left,center,S,L,T,r,call,iteration-1)
    else:
        return Bisection2(center,right,S,L,T,r,call,iteration-1)
    
Q3Newton=[]
def Newton(initsigma,S,L,T,r,call,iteration):
    sigma = initsigma
    for i in range(iteration):
        fx = blscall(S,L,T,r,sigma)-call
        Q3Newton.append(sigma)
        fx2 = (blscall(S,L,T,r,sigma+0.00000001)-blscall(S,L,T,r,sigma-0.00000001))/0.00000002;
        sigma = sigma - fx/fx2
    return sigma

S = 10326.68
L = 10300.0
T = 21.0/365
r = 0.01065
sigma = 0.10508

print(blscall(S,L,T,r,sigma))
sigma = Bisection(0.0000001,1,S,L,T,r,121.0,0.000000001)
print(sigma)
print(blscall(S,L,T,r,sigma))
print(Newton(0.5,S,L,T,r,121.0,20))
print(Bisection2(0.00001,1.0,S,L,T,r,121.0,20))
plt.plot(Q3Bi)
plt.plot(Q3Newton)




