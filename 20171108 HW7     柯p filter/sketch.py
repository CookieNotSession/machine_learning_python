#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 20:45:32 2017

@author: cookie040667
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 19:08:14 2017

@author: cookie040667
"""
#gaussion blur hw1
#motion blur hw2
#sharp filter hw3
# sobelx sobley -> Ix^2+Iy^2 拉成一維矩陣 把20%以下標黑色 以上標白色
import numpy as np
from PIL import Image
from scipy import signal #數學矩陣運算lib
def conv2(I,M): #二維(非彩色)
    IH,IW = I.shape
    MH,MW = M.shape
    out = np.zeros((IH-MH+1,IW-MW+1)) #新的長寬
    for h in range(IH-MH+1):
        for w in range(IW-MW+1):
            for y in range(MH):
                for x in range(MW):
                    out[h,w] = out[h,w]+I[h+y,w+x]*M[y,x] #一張照片做mask處理
    return out           
    
I = Image.open('photo.jpg')
data = np.asanyarray(I)
#I.show()
#data = np.asanyarray(I)[:,:,::-1]
#data2 = np.zeros((900,900,3)).astype('uint8')
#data2[:,:,1] = data[:,:,1] 
#data2 = 255 - data #0變255 255變0
#M = np.ones((1,60))/60 #5*5矩陣 全變1/25 motion blur 作業1
#x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
#d = np.sqrt(x*x+y*y)
#sigma, mu = 1.0, 0.0
#Mbig = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )/100

sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) 
#sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
R = data[:,:,0]
R2 = signal.convolve2d(R,sobelx,boundary='symm',mode='same')
G = data[:,:,1]
G2 = signal.convolve2d(G,sobelx,boundary='symm',mode='same')
B = data[:,:,2] #紅綠藍分開做
B2 = signal.convolve2d(B,sobelx,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')


sobely = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
R = data[:,:,0]
R3 = signal.convolve2d(R,sobely,boundary='symm',mode='same')
G = data[:,:,1]
G3 = signal.convolve2d(G,sobely,boundary='symm',mode='same')
B = data[:,:,2] #紅綠藍分開做
B3 = signal.convolve2d(B,sobely,boundary='symm',mode='same')
data3 = data.copy()
data3[:,:,0] = R3.astype('uint8')
data3[:,:,1] = G3.astype('uint8')
data3[:,:,2] = B3.astype('uint8')

Graph = data2**2 + data3**2
#newgraph = Image.fromarray(Graph,'RGB')
#newgraph.show()

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

newgraygraph = rgb2gray(Graph)
D = []
for i in range(900):
    for j in range(900):
        D.append(newgraygraph[i,j])
band = int(900*900*0.6) #第幾小的數字
bandindex = np.argsort(D)[band-1] #找第486000筆小的index
standard = newgraygraph[bandindex//900,bandindex%900-1] #x列的數字

for i in range(900):
    for j in range(900):
        if(newgraygraph[i,j]>=standard):
            Graph[i,j] = np.array([0,0,0])
        else:
            Graph[i,j] = np.array([255,255,255])
I4 = Image.fromarray(Graph,'RGB')
I4.show()

        
        
#x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
#d = np.sqrt(x*x+y*y)
#sigma, mu = 0.5, 0.0
#Msmall = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )/100
#
#M= Mbig - Msmall
"""
R = data[:,:,0]
R2 = signal.convolve2d(R,M,boundary='symm',mode='same')
G = data[:,:,1]
G2 = signal.convolve2d(G,M,boundary='symm',mode='same')
B = data[:,:,2] #紅綠藍分開做
B2 = signal.convolve2d(B,M,boundary='symm',mode='same')
data2 = data.copy()
data2[:,:,0] = R2.astype('uint8')
data2[:,:,1] = G2.astype('uint8')
data2[:,:,2] = B2.astype('uint8')

I2 = Image.fromarray(data2,'RGB')
I2.show()
"""