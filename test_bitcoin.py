# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:45:44 2020

@author: hexie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 20:24:54 2020

@author: hexie
"""

import numpy as np
import pandas as pd
import OLP as olp 
import matplotlib.pyplot as plt
edges_orig = []
path = r"C:\Users\hexie\Desktop\multilayer_link\OptimalLinkPrediction-master\OptimalLinkPrediction-master\Code\data\bitcoin\\"
r_path = r"C:\Users\hexie\Desktop\multilayer_link\OptimalLinkPrediction-master\OptimalLinkPrediction-master\Code\results\bitcoin\\"

data_length= 20
layer_of_predict =[5,6,7,8]
for i in range(1,31):
    edges_orig.append(np.loadtxt(path+"bitcoin_{}.txt".format(i)))

AUC_all = []
Precision_all = []
Recall_all=[]
#%%

# how many layers do we go back to the search
layer_of_predict =[5,6,7,8]
# number of layer used in the search
predict_num = [2,3,4,5]

layer_pair = 4

# which time slot we predict on
predicted_layers = [8,9,10,11,12]

for j in range(layer_pair):
    AUC_temp = []
    Precision_temp = []
    Recall_temp=[]
    for i in predicted_layers:
        auc, precision, recall = olp.topol_stacking_temporal_3(edges_orig[i-layer_of_predict[j]:i],edges_orig[i],predict_num[j])
        AUC_temp.append(auc)
        Precision_temp.append(precision)
        Recall_temp.append(recall)
        print("finished doing predicted_layers", i, layer_of_predict[j], predict_num[j])
    AUC_all.append(AUC_temp)
    Precision_all.append(Precision_temp)
    Recall_all.append(Recall_temp)
#%%
#x = olp.topol_stacking_temporal(edges_orig[0:2],edges_orig[2])
result = np.array((AUC_all, Precision_all, Recall_all), dtype=float)
np.save(r_path+"Bitcoin_test_result.npy",result)
x=predicted_layers
colors = "bgrcmykw"
color_index = 0

for i in range(len(AUC_all)):
    plt.scatter(x, AUC_all[i], alpha=0.8,c=colors[i],label = "layer_"+str(predicted_layers[i]))

plt.legend(loc='lower right', ncol=5, fontsize=8) 
plt.title("AUC score for bitcoin")
plt.savefig(r_path+'bitcoin.png', bbox_inches='tight')
plt.show()