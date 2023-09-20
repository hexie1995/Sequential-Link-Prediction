import pickle
import numpy as np
import OLP_FINAL_PARTIAL as tolp
from multiprocessing import Pool
path = r"/home2/xhe/new_fake/new_fake1//"
fakelist1 = []
f1 = ["1","2","3"]
f2 = ["1","2","5","0"]

for i in f1:
    for j in f1:
        for k in f2:
            fakelist1.append("fake"+str(i)+str(j)+str(k))

fakelist2 = []

for i in f1:
    for j in f1:
        fakelist2.append("fake"+str(i)+str(j)+str(15))

fakelist = fakelist1+fakelist2


fakepath = r"/home/xhe/TOP_FINAL_2_PARTIAL/FAKE1/fake1_full_results//"
auc_all_new1 = []
opt_auc_new = []
for fa in fakelist:
    l1 = np.load(fakepath + "AUC_" + fa + ".npy").tolist()
    auc_all_new1.append(l1)
    
auc_all_old = np.array(auc_all_new1)
top = auc_all_old[:,0]
ts = auc_all_old[:,1]
mlsbm = auc_all_old[:,2]
elstmd = auc_all_old[:,3]
topall = auc_all_old[:,4]

auc_all = np.array(auc_all_old)

auc_dict = dict(zip(fakelist, auc_all))
print(auc_dict)