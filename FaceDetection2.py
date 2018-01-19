# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 18:45:56 2017

@author: USER
"""
"""
import numpy as np
import matplotlib.pyplot as plt
import math
npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']
#用4個array將npz的值取回

trpn = trainface.shape[0]
trnn = trainnonface.shape[0]
tepn = testface.shape[0]
tenn = testnonface.shape[0]

fn = 0
ftable = [] #建立一張可以找回原圖的table this is a 36648*5 Matrix
for y in range(19):
    for x in range(19):
        for h in range(2,20): #2到最大19 最大視窗容許佔滿整個螢幕
            for w in range(2,20):
                if(y+h<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([0,y,x,h,w]) #每一個合法特徵 就把yxhw留下 0代表第0類的方塊
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w<=19):
                    fn = fn + 1
                    ftable.append([1,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h<=19 and x+w*3<=19):
                    fn = fn + 1
                    ftable.append([2,y,x,h,w])
for y in range(19):
    for x in range(19):
        for h in range(2,20):
            for w in range(2,20):
                if(y+h*2<=19 and x+w*2<=19):
                    fn = fn + 1
                    ftable.append([3,y,x,h,w])

#sample is N-by-361 matrix  / sample:所有臉 非人臉照片 ftable : 36648*5 Matrix (給我一個c 取c的特徵)
# return a vector with N feature values
def fe(sample,ftable,c): #feature extraction: 給我一個N*361的矩陣 給我ftable與某個特徵 我就把N*1算出來給你 N個人的全部特徵全算出來
    ftype = ftable[c][0] 
    y = ftable[c][1]
    x = ftable[c][2]
    h = ftable[c][3]
    w = ftable[c][4]
    T = np.arange(361).reshape((19,19)) #準備0-360的一維矩陣: 
    if(ftype==0): #flatten拉成一維 把sample裡面的每一個row的23 21 22col取出來 做一次sum 往右加上去 (將2400多人的白色區域 一次算完)
        #開好一張table 可以用來查詢到底我現在有感興趣的區塊 在361中屬於哪些格子 去這張表中把對應到區塊的index拿出 並拉成一維 
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)
        #我在我原始的sample可能是face(2000*361)的sample 這2000多人我全都要 而他的column我取23 21 22拿出來
        #我取完這些之後 就是一個2000多*4的矩陣 然後把他對1軸 sum起來 就會成為2000多*1的矩陣
    if(ftype==1):
        output = -np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)
    if(ftype==2):
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)+np.sum(sample[:,T[y:y+h,x+w*2:x+w*3].flatten()],axis=1)
    if(ftype==3):
        output = np.sum(sample[:,T[y:y+h,x:x+w].flatten()],axis=1)-np.sum(sample[:,T[y:y+h,x+w:x+w*2].flatten()],axis=1)-np.sum(sample[:,T[y+h:y+h*2,x:x+w].flatten()],axis=1)+np.sum(sample[:,T[y+h:y+h*2,x+w:x+w*2].flatten()],axis=1)
    return output

#把人臉2400張照片 建立一個矩陣2400*36648 每一張照片就會有36648個特徵值去描述他
trpf = np.zeros((trpn,fn)) #2400*36648
trnf = np.zeros((trnn,fn)) 

#---------------------
#tepf = np.zeros((tepn,fn))
#tenf = np.zeros((tenn,fn))
#--------------------

for c in range(fn): 
    trpf[:,c] = fe(trainface,ftable,c) #第c個column 把trainface的資料庫(2400*36648)丟入 就能取出第c個特徵值
    trnf[:,c] = fe(trainnonface,ftable,c)
#簡單的分類方法 在任何一地方砍一刀theta 去決定左右邊哪邊是人臉 針對每一個特徵 都去做這件事情 找到他的最佳解
#砍一刀決定左右 實作就是到處try 看哪一刀最好 但如果做的更精細 更好 --> weak leaner / weak classifier

#----------AdaBoost 驗算法開始！ 
#我把2400 4000張照片 共6000張照片 放在一起    
    
pw = np.ones((trpn,1))/trpn/2 #positive weight 
nw = np.ones((trnn,1))/trnn/2
#用兩個矩陣 與 這倆個權重 開始

def WC(pw,nw,pf,nf): #Weak Classifier 傳入正負資料的weight 與 正負資料的特徵  
   #特徵值最大 最小取出來 分為十等分 
    maxf = max(pf.max(),nf.max()) #正特徵的最大值 與 負特徵最大值
    minf = min(pf.min(),nf.min())
    theta = (maxf-minf)/10+minf #從左邊數來 十分之一的地方 割下第一刀 
    error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])#所有比他還要小的
    polarity = 1 #這個情況都是1 
    if(error>0.5): #如果error超過0.5 反過來猜 就是把polarity改為0
        error = 1-error 
        polarity = 0
    min_theta = theta
    min_error = error
    min_polarity = polarity
    for i in range(2,10): #從第2刀到第9刀都去檢查有沒有比他的min還小
        theta = (maxf-minf)*i/10+minf
        error = np.sum(pw[pf<theta])+np.sum(nw[nf>=theta])
        polarity = 1
        if(error>0.5):
            error = 1-error
            polarity = 0
        if(error<min_error): #如果這次error小於min_error 才可更新
            min_theta = theta
            min_error = error
            min_polarity = polarity
    return min_error,min_theta,min_polarity

SC = []
for t in range(10): #跑10個特徵
    weightsum = np.sum(pw)+np.sum(nw) #把pw nw normalize過 (要先做過！不然後面weightsum會不同)
    pw = pw/weightsum
    nw = nw/weightsum
    best_error,best_theta,best_polarity = WC(pw,nw,trpf[:,0],trnf[:,0]) #train postive feature第0個特徵
    #呼叫WC 給他目前的pn nw 與 正data(2400多) 負data(4500多)的0號特徵 丟進去給他
    #_________________________________________
    #把所有的值域 等切成10刀 用這10刀決定右+左-之標準 算出一個error 如果這error太大(大於0.5)就反過來猜 polarity換邊
    best_feature = 0 #先假設最好的是0號特徵
    for i in range(1,fn): #跑3萬多個 只要有人error比較小就把他蓋掉
        error,theta,polarity = WC(pw,nw,trpf[:,i],trnf[:,i]) #算出這組最佳解與其error多少
        if(error<best_error): #代表這組解為最佳解
            best_feature = i #把最棒的特徵 改成現在這個第i個特徵
            best_error = error
            best_theta = theta
            best_polarity = polarity
    #------目前已找到最小error他的分法 Global Optimalization出現了 36648筆資料中最好的那個人 把他的best error拿來算alpha beta
    beta = best_error/(1-best_error) #分對者乘以beta
    alpha = math.log10(1/beta)
    SC.append([best_feature,best_theta,best_polarity,alpha]) #用哪些特徵(best feature) 
    if(best_polarity==1): #-----把分對的人 權重乘以beta(一個小於1的值) 因此分對的權重會縮小
        #分對的指 : polarity==1者 代表右+左- 因此只要大於theta的正data 全都要乘上beta 因為他們分對
        #而小於theta的那些負data也要乘以beta
        pw[trpf[:,best_feature]>=best_theta] = pw[trpf[:,best_feature]>=best_theta]*beta
        nw[trnf[:,best_feature]<best_theta] = nw[trnf[:,best_feature]<best_theta]*beta
    else: #所有小於的正data 才乘beta 大於等於的負data 乘beta
        pw[trpf[:,best_feature]<best_theta] = pw[trpf[:,best_feature]<best_theta]*beta
        nw[trnf[:,best_feature]>=best_theta] = nw[trnf[:,best_feature]>=best_theta]*beta
#------------------------------------目前已經取出一組特徵       
trps = np.zeros((trpn,1)) #train positive score
trns = np.zeros((trnn,1)) #train negative score
alpha_sum = 0
for i in range(10):
    feature = SC[i][0] 
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum = alpha_sum + alpha
    if(polarity==1):#代表右+左-
        trps[trpf[:,feature]>=theta] = trps[trpf[:,feature]>=theta]+alpha #2400*36648 把他的第i個特徵取出 若他為正 就說他大於等於theta 他得分了 因此自己加上了alpha 得了alpha分
        trns[trnf[:,feature]>=theta] = trns[trnf[:,feature]>=theta]+alpha 
    else:
        trps[trpf[:,feature]<theta] = trps[trpf[:,feature]<theta]+alpha 
        trns[trnf[:,feature]<theta] = trns[trnf[:,feature]<theta]+alpha

trps = trps/alpha_sum #把算出來的總分(10個alpha) 如果全都對 算出來會是1 只要大於0.5就說對的
trns = trns/alpha_sum
"""
#-----------------------------------------
test = testface[5].reshape((1,361))
teps = np.zeros((1,1))
testpf = np.zeros((1,fn))
for c in range(fn):
    testpf[:,c] = fe(test,ftable,c)
alpha_sum = 0
for i in range(20):
    feature = SC[i][0]
    theta = SC[i][1]
    polarity = SC[i][2]
    alpha = SC[i][3]
    alpha_sum = alpha_sum + alpha
    if(polarity==1):
        teps[testpf[:,feature]>=theta] = teps[testpf[:,feature]>=theta]+alpha
    else:
        teps[testpf[:,feature]<theta] = teps[testpf[:,feature]<theta]+alpha

teps = teps/alpha_sum
