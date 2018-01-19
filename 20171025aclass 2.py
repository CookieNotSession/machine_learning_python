#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:59:48 2017

@author: cookie040667
"""

import pandas as pd
df = pd.read_csv('TXF20112015.csv',sep=',',header=None)
TAIEX = df.values
tradeday = list(set(TAIEX[:,0]//10000)) #把日期欄位取出 除10000 set:重複的去掉 
tradeday.sort() #list做完 對list做sort