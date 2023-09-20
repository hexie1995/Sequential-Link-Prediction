# +
import numpy as np
from TOLPTS import *
from multiprocessing import Pool

path = r"/home2/xhe/fake2//"
fakelist1 = []
fakelist2 = []
f1 = ["1","2","3"]
f2 = ["1","2","5","0"]

for i in f1:
    for j in f1:
        for k in f2:
            fakelist1.append("fake"+str(i)+str(j)+str(k))

for i in f1:
    for j in f1:
        fakelist2.append("fake"+str(i)+str(j)+str(15))


fakelist = fakelist1+fakelist2



# -

def run_ts(name):
    edges_orig = [] 
    data_length= 7

    for i in range(1,data_length+1):
        edges_orig.append(np.loadtxt(path+name+"/"+ name+"_{}.txt".format(i)))

    target_layer = edges_orig[6]
    edges_orig = edges_orig[0:6]
    predict_num = 3
    
    X_timeseries, y_timeseries = calculate_ts_scores(edges_orig, target_layer, predict_num,name)
    
    np.save("time_series_matrix/X_ts_"+(name)+".npy",X_timeseries)
    np.save("time_series_matrix/y_ts_"+(name)+".npy",y_timeseries)
    
    AUPRC = []
    AUC = []
    PRE = []
    REC = []
    CM = []
    for i in range(42):
        auprc, auc, precision, recall, cm = rf_with_chosen_feats(X_timeseries, y_timeseries, i)
        AUPRC.append(auprc)
        AUC.append(auc)
        REC.append(recall)
        CM.append(cm)
 
    auprc, auc, precision, recall, cm = rf_with_chosen_feats(X_timeseries, y_timeseries, 100)
    AUPRC.append(auprc)
    AUC.append(auc)
    REC.append(recall)
    CM.append(cm)



    np.save("time_series_results/AUPRC_"+(name)+".npy",AUPRC)
    np.save("time_series_results/AUC_"+(name)+".npy",AUC)
    np.save("time_series_results/PRE_"+(name)+".npy",PRE)
    np.save("time_series_results/REC_"+(name)+".npy",REC)
    np.save("time_series_results/CM_"+(name)+".npy",CM)

with Pool(len(fakelist)) as p:
    print(p.map(run_ts, fakelist))

#run_ts(fakelist[0])


