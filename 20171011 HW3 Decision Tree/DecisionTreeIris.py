# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 19:50:58 2017

@author: USER
"""

import math
import numpy as np
from sklearn import datasets
data = datasets.load_iris()

       
def entropy(p1,n1):
    if(p1==0 and n1==0):
        return 1
    value = 0
    pp = p1/(p1+n1)
    pn = n1/(p1+n1)
    if(pp>0):
        value -= pp*math.log2(pp)
    if(pn>0):
        value -= pn*math.log2(pn)
    return value
    
def IG(p1,n1,p2,n2):
    num = p1+n1+p2+n2
    num1 = p1+n1
    num2 = p2+n2
    return entropy(p1+p2,n1+n2)-num1/num*entropy(p1,n1)-num2/num*entropy(p2,n2)


def recover(Tree,tt,h):
    test_feature = data['data'][tt,:]
    now = 0
    while(Tree[now]['leaf']==0):
        bestf = Tree[now]['selectf']
        thres = Tree[now]['threshold']
        if(test_feature[bestf]<=thres):
            now = Tree[now]['child'][0]
        else:
            now = Tree[now]['child'][1]
        #print(target[i],Tree[now]['decision'])
    if(h=='01'):
        return Tree[now]['decision']
    if(h=='12'):
        return Tree[now]['decision']+1
    if(h=='02'):
        if Tree[now]['decision']==0:
            return Tree[now]['decision']
        if Tree[now]['decision']==1:
            return Tree[now]['decision']+1
        
        
def DT(skip,h):
    
    if(skip==0):#拿第0筆(target:0)資料當作test
        if(h=='01'):
            target=data['target'][1:100] #共49+50筆
            feature=data['data'][1:100] 
        elif(h=='12'):
            target=data['target'][50:150]-1 #共50+50筆
            feature=data['data'][50:150] 
        else: #02
            target1=data['target'][1:50] #49+50筆
            target2=data['target'][100:150]-1
            target=np.concatenate((target1,target2),axis=0)
            
            feature1=data['data'][1:50]
            feature2=data['data'][100:150]
            feature=np.concatenate((feature1,feature2),axis=0)
    elif(skip>0 and skip<=49): #輪流拿1~49 target為0的資料去做test
        if(h=='01'):
            target1=data['target'][0:skip]
            target2=data['target'][skip+1:100]
            target=np.concatenate((target1,target2), axis=0)
            
            feature1=data['data'][0:skip]
            feature2=data['data'][skip+1:100]
            feature=np.concatenate((feature1,feature2),axis=0)
        elif(h=='12'):
            target=data['target'][50:150]-1
            feature=data['data'][50:150]
        else: #02
            target1=np.concatenate((data['target'][0:skip],data['target'][skip+1:50]), axis=0)
            target2=data['target'][100:150]-1
            target=np.concatenate((target1,target2),axis=0)
            
            feature1=np.concatenate((data['data'][0:skip],data['data'][skip+1:50]),axis=0)
            feature2=data['data'][100:150]
            feature=np.concatenate((feature1,feature2),axis=0)
    elif(skip>=50 and skip<=99):
            if(h=='01'):
                target1=data['target'][0:skip]
                target2=data['target'][skip+1:100]
                target=np.concatenate((target1,target2), axis=0)
                
                feature1=data['data'][0:skip]
                feature2=data['data'][skip+1:100]
                feature=np.concatenate((feature1,feature2),axis=0)
            elif(h=='12'):
                target1=np.concatenate((data['target'][50:skip]-1,data['target'][skip+1:100]-1),axis=0)
                target2=data['target'][100:150]-1
                target=np.concatenate((target1,target2),axis=0)
                feature1=np.concatenate((data['data'][50:skip],data['data'][skip+1:100]),axis=0)
                feature2=data['data'][100:150]
                feature=np.concatenate((feature1,feature2),axis=0)
            else:#h==02
                target1=data['target'][0:50]
                target2=data['target'][100:150]-1
                target = np.concatenate((target1,target2),axis=0)
                feature1=data['data'][0:50]
                feature2=data['data'][100:150]
                feature = np.concatenate((feature1,feature2),axis=0)
    elif(skip>=100 and skip<=149): 
        if(h=='01'):
            target=data['target'][0:100]
            feature=data['data'][0:100]
        elif(h=='12'):
            target1=data['target'][50:100]-1
            target2=np.concatenate((data['target'][100:skip]-1,data['target'][skip+1:149]-1),axis=0)
            target=np.concatenate((target1,target2),axis=0)
            feature1=data['data'][50:100]
            feature2=np.concatenate((data['data'][100:skip],data['data'][skip+1:149]),axis=0)
            feature=np.concatenate((feature1,feature2),axis=0)
        else:#h==02
            target1=data['target'][0:50]
            target2=np.concatenate((data['target'][100:skip]-1,data['target'][skip+1:150]-1),axis=0)
            target=np.concatenate((target1,target2),axis=0)
            feature1=data['data'][0:50]
            feature2=np.concatenate((data['data'][100:skip],data['data'][skip+1:150]),axis=0)
            feature=np.concatenate((feature1,feature2),axis=0)
            
    
    #data = np.loadtxt('PlayTennis.txt',usecols=range(5),dtype=int)
    #feature = data[:,0:4]
    #target = data[:,4]-1
    
    node = dict()
    node['data'] = range(0,len(target))
    Tree = []
    Tree.append(node)
    t = 0
    while(t<len(Tree)):
        idx = Tree[t]['data']
        if(sum(target[idx])==0):
            Tree[t]['leaf']=1
            Tree[t]['decision']=0
        elif(sum(target[idx])==len(idx)):
            Tree[t]['leaf']=1
            Tree[t]['decision']=1
        else:
            bestIG = 0
            for i in range(feature.shape[1]):
                pool = list(set(feature[idx,i]))
                pool.sort()
                for j in range(len(pool)-1):
                    thres = (pool[j]+pool[j+1])/2
                    G1 = []
                    G2 = []
                    for k in idx:
                        if(feature[k,i]<=thres):
                            G1.append(k)
                        else:
                            G2.append(k)
                    thisIG = IG(sum(target[G1]==1),sum(target[G1]==0),sum(target[G2]==1),sum(target[G2]==0))
                    if(thisIG>bestIG):
                        bestIG = thisIG
                        bestG1 = G1
                        bestG2 = G2
                        bestthres = thres
                        bestf = i
            if(bestIG>0):
                Tree[t]['leaf']=0
                Tree[t]['selectf']=bestf
                Tree[t]['threshold']=bestthres
                Tree[t]['child']=[len(Tree),len(Tree)+1]
                node = dict()
                node['data']=bestG1
                Tree.append(node)
                node = dict()
                node['data']=bestG2
                Tree.append(node)
            else:
                Tree[t]['leaf']=1
                if(sum(target(idx)==1)>sum(target(idx)==0)):
                    Tree[t]['decision']=1
                else:
                    Tree[t]['decision']=0
        t+=1
    return recover(Tree,tt,h)
    
true=0
false=0
    
for tt in range(150):
    cal=[0,0,0]
    cal[DT(tt,'01')]+=1
    cal[DT(tt,'12')]+=1
    cal[DT(tt,'02')]+=1
    
    if (cal[0]==1 and cal[1]==1):
        newprediction=0
    elif(cal[0]>=2):
        newprediction=0 #預測是setosa
    elif(cal[1]>=2):
        newprediction=1 #預測是versicolor
    elif(cal[2]>=2):
        newprediction=2 #預測是virginica
    if(data['target'][tt]==newprediction): #預測與真實比對
        true=true+1
        print('true',tt)
    else:
        false=false+1
        print('false',tt)
print('True:',true)
print('False:',false)
print('Error:',false/(true+false))  #錯誤率
 
