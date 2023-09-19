import numpy as np
import TOLP as tolp
from multiprocessing import Pool
path = r"community_label_TSBM//"

data_name = "fake110"


def completely_unobserved_data(name):
    edges_orig = [] 
    data_length = 8
    
    
    for i in range(1,data_length):
        edges_orig.append(np.loadtxt(path+name+"/"+ name+"_{}.txt".format(i)))

    lstm = np.load(path+name+"/"+ name+".npy")
    target_layer = edges_orig[6]
    edges_orig = edges_orig[0:6]
    predict_num = 3
    auprc, auc, precision, recall, featim, feats = tolp.topol_stacking_temporal_with_edgelist(edges_orig, target_layer, predict_num,name)
    
    print("This is the Revision 1 AUC score")
    print(auc)

    

completely_unobserved_data(data_name)

def partially_observed_data(name):
    edges_orig = [] 
    data_length = 8
    
    
    for i in range(1,data_length):
        edges_orig.append(np.loadtxt(path+name+"/"+ name+"_{}.txt".format(i)))

    lstm = np.load(path+name+"/"+ name+".npy")
    target_layer = edges_orig[6]
    edges_orig = edges_orig[0:6]
    predict_num = 3
    auprc, auc, precision, recall, featim, feats = tolp.topol_stacking_temporal_partial(edges_orig, target_layer, predict_num,name)
    
    print("This is the Revision 1 AUC score")
    print(auc)
    return auc
