import pickle
import numpy as np
import OLP_FINAL_PARTIAL as tolp
from multiprocessing import Pool
path = r"/home/xhe/real_data//"


data_list1 = ["chess","obrazil","bionet1", "bitcoin","emaildnc","bionet2",
              "obitcoin","london","collegemsg","fbmsg","radoslaw", "fbforum", "mit",
             "ant1","ant2","ant3","ant4","ant5","ant6"]

def run_data(name):
    edges_orig = [] 
    data_length= 8

    for i in range(1,data_length):
        edges_orig.append(np.loadtxt(path+name+"/"+ name+"_{}.txt".format(i)))

    lstm = np.load(path+name+"/"+ name+".npy")
    predict_num = 3
    auprc, auc, precision, recall, featim, feats = topol_stacking_temporal_with_edgelist(edges_orig[0:6],edges_orig[6], predict_num, name, lstm)
    

     
with Pool(len(data_list1)) as p:
    print(p.map(run_data, data_list1))

#run_data("bitcoin")