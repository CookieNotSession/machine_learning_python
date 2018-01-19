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
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
d = np.sqrt(x*x+y*y)
sigma, mu = 1.0, 0.0
M = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )/100

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