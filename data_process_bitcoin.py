# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 18:16:07 2020

@author: hexie
"""

import pandas as pd
import numpy as np
data = pd.read_csv("./data/soc-sign-bitcoinotc.csv",header=None)
# number of rows in the data
n = data.shape[0] 
# divide by 100 to get 100 slice of the data at different time window
count = int(n/100)
# generate edgelist in the form of the orignial code to simplify the process
start = 0
for i in range(1,120):
    x= count*i
    save = data.iloc[start:x, 0:2]
    np.savetxt(r'C:\Users\hexie\Desktop\multilayer_link\OptimalLinkPrediction-master\OptimalLinkPrediction-master\Code\data\Abitcoin\Abitcoin_{}.txt'.format(i), save.values)
    #start = x