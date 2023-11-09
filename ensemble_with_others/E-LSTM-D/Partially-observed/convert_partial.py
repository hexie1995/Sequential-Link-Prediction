from model import e_lstm_d
import os
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
from utils import *
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import tensorflow.keras.backend as K
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

path = r"/../../community_label_TSBM//"

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
data_list = ["chess","obrazil","bionet1", "bitcoin","emaildnc","bionet2",
              "obitcoin","london","collegemsg","fbmsg","radoslaw", "fbforum", "mit",
             "ant1","ant2","ant3","ant4","ant5","ant6"]



def mxx(edges_orig):
    n_layers = len(edges_orig)
    temp_max = np.max(edges_orig[0])
    for i in range(n_layers):
        temp_max = max(temp_max,np.max(edges_orig[i]))
    return temp_max

def gen_tr_ho_networks(A_orig, alpha, alpha_):

    """ 
    This function constructs the holdout and training adjacency matrix uniformly sampled from the original adjacency matrix.

    Input and Parameters:
    -------
    A_orig: the original adjacency matrix

    Returns:
    -------
    A_ho : the holdout adjacency matrix
    A_tr : training adjacency matrix
    alpha : the sample probability used for sampling from the original matrix to create the holdout matrix
    alpha_ : the sample probability used for sampling from the holdout matrix to create the training matrix

    Examples:
    -------
    >>> A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    """

    A_ho = 1*(np.triu(A_orig,1)==1)
    rows_one, cols_one = np.where(np.triu(A_ho,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha)
    A_ho[rows_one, cols_one] = ones_prob_samp
    A_ho = A_ho + A_ho.T

    A_tr = 1*(np.triu(A_ho,1)==1)
    rows_one, cols_one = np.where(np.triu(A_tr,1))
    ones_prob_samp = np.random.binomial(1, size=len(rows_one), p=alpha_)
    A_tr[rows_one, cols_one] = ones_prob_samp
    A_tr = A_tr + A_tr.T
    return A_ho, A_tr

def convert_to_npy(name):
    edges_orig = [] 
    data_length= 8

    for i in range(1,data_length):
        edges_orig.append(np.loadtxt(path+name+"/"+ name +"_{}.txt".format(i)))
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    
    
    
    #### load target layer A
    
    num_nodes = int(mxx(edges_orig))+1
    
    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense() 
    
    
    A = A_tr[-1]
    
    A_return = A_tr[0:6]
    
    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix

    ### here we are still missing the last training and testing label, we need to add the target layer
    A_hold_, A_train_ = gen_tr_ho_networks(A, alpha, alpha_)    
    A_return.append(A_train_)
    ### A_ho is the test sets, the label is. The last element of A_ho is the true label where we try to predict. 
    A_return.append(A_hold_)    
    np.save(path+name+"/"+ name +"_partial.npy",A_return)
    
    return num_nodes
    
data_name = "fake110"
convert_to_npy(data_name)
#node_count = []
#for item in data_list:
#    node_count.append(convert_to_npy(item))
