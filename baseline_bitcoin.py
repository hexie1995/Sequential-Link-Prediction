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
for i in range(1,31):
    edges_orig.append(np.loadtxt(path+"bitcoin_{}.txt".format(i)))

AUC_all = []
Precision_all = []
Recall_all=[]
#%%
# which time slot we predict on
predicted_layers = [8,9,10,11,12]

for i in predicted_layers:
    auc, precision, recall = olp.topol_stacking(edges_orig[i])
    AUC_all.append(auc)
    Precision_all.append(precision)
    Recall_all.append(recall)
    print("finished doing predicted_layers", i)

#%%
#x = olp.topol_stacking_temporal(edges_orig[0:2],edges_orig[2])
result = np.array((AUC_all, Precision_all, Recall_all), dtype=float)
np.save(r_path+"Bitcoin_basline.npy",result)
x=predicted_layers

plt.scatter(x, AUC_all, alpha=0.8,c="red",label="baseline")

plt.legend(loc='lower right', ncol=5, fontsize=8) 
plt.title("Baseline AUC score for bitcoin")
plt.savefig(r_path+'bitcoin_basline.png', bbox_inches='tight')
plt.show()