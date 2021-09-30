from utils import clearCache
clearCache()
import numpy as np
import OLP_new as olpn
from multiprocessing import Pool
path = r"/nas/longleaf/home/hexie/OptimalLinkPrediction-master/OptimalLinkPrediction-master/real_data//"

data_list1 = ["bionet1","bionet2","chess", "bitcoin","collegemsg","obitcoin","obrazil","radoslaw", "london", "mit"]
data_list2 = ["ant1","ant2","ant3","ant4","ant5","ant6"]
data_list3 = ["fbforum", "fbmsg", "emaildnc"]
data_list = data_list1 + data_list2 + data_list3


# +
def run_data(name):
    edges_orig = [] 
    data_length= 8

    for i in range(1,data_length):
        edges_orig.append(np.loadtxt(path+name+"/"+ name+"_{}.txt".format(i)))

    lstm = np.load(path+name+"/"+ name+".npy")
    target_layer = edges_orig[6]
    edges_orig = edges_orig[0:6]
    predict_num = 3
    
    out_order, auprc, auc, precision, recall , featim = olpn.topfinal(edges_orig[0:6],edges_orig[6], predict_num, name, lstm)
    np.savetxt(str(name)+"_auc.txt", auc)
    np.savetxt(str(name)+"_auprc.txt", auprc)
    np.savetxt(str(name)+"_precision.txt", precision)
    np.savetxt(str(name)+"_recall.txt", recall)
    np.savetxt(str(name)+"_featim.txt", featim)
    
    
# -

for item in data_list:
    run_data(item)
