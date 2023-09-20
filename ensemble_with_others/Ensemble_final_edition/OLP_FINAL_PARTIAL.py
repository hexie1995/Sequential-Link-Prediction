# -*- coding: utf-8 -*-
from __future__ import division
import os
import copy
import os.path
import random
from scipy import linalg
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import SelectFromModel
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
import time
import math
from random import randint
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


""
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

def adj_to_nodes_edges(A):
    
    """ 
    This function change adjacency matrix to list of nodes and edges.

    Input and Parameters:
    -------
    A: the adjacency matrix

    Returns:
    -------
    nodes: node list of the given network
    edges: edge list of the given network

    Examples:
    -------
    >>> nodes, edges = adj_to_nodes_edges(A)
    """
    
    num_nodes = A.shape[0]
    nodes = range(num_nodes)
    edges = np.where(np.triu(A,1))
    row = edges[0]
    col = edges[1]
    edges = np.vstack((row,col)).T
    return nodes, edges


def sample_true_false_edges(A_orig, A_tr, A_ho):  
    
    """ 
    This function creates the training and holdout samples.

    Input and Parameters:
    -------
    A: the adjacency matrix

    Returns:
    -------
    nodes: node list of the given network
    edges: edge list of the given network

    Examples:
    -------
    >>> nodes, edges = adj_to_nodes_edges(A)
    """
    
    nodes, edge_tr = adj_to_nodes_edges(A_tr)
    nsim_id = 0
    np.random.seed(nsim_id)

    A_diff = A_ho - A_tr
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_ho_aux = -1*A_ho + 1
    ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
    Nsamples = 2000 # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
	    edge_t_idx_aux = np.random.randint(len(e_diff[0]))
	    edge_f_idx_aux = np.random.randint(len(ne_ho[0]))
	    edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
	    edge_f.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux]))
	
	# store for later use
    if not os.path.isdir("./edge_tf_tr/"):
	    os.mkdir("./edge_tf_tr/")
    np.savetxt("./edge_tf_tr/edge_t.txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_tr/edge_f.txt",edge_f,fmt='%u')
    
    A_diff = A_orig - A_ho
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_tt = [] # list of true edges (positive samples)
    edge_ff = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
        edge_tt_idx_aux = np.random.randint(len(e_diff[0]))
        edge_ff_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_tt.append((e_diff[0][edge_tt_idx_aux],e_diff[1][edge_tt_idx_aux]))
        edge_ff.append((ne_orig[0][edge_ff_idx_aux],ne_orig[1][edge_ff_idx_aux]))
    
    # store for later use
    if not os.path.isdir("./edge_tf_ho/"):
        os.mkdir("./edge_tf_ho/")
    np.savetxt("./edge_tf_ho/edge_t.txt",edge_tt,fmt='%u')
    np.savetxt("./edge_tf_ho/edge_f.txt",edge_ff,fmt='%u')
    return edge_t, edge_f, edge_tt, edge_ff

def gen_topol_feats(A_orig, A, edge_s): 
    
    """ 
    This function generates the topological features for matrix A (A_tr or A_ho) over edge samples edge_s (edge_tr or edge_ho).

    Input and Parameters:
    -------
    A: the training or holdout adjacency matrix that the topological features are going to be computed over
    A_orig: the original adjacency matrix
    edge_s: the sample set of training or holdout edges that the topological features are going to be computed over

    Returns:
    -------
    df_feat: data frame of features

    Examples:
    -------
    >>> gen_topol_feats(A_orig, A_tr, edge_tr)
    >>> gen_topol_feats(A_orig, A_ho, edge_ho)
    """
    
    _, edges = adj_to_nodes_edges(A)    
    nodes = [int(iii) for iii in range(A.shape[0])]
    N = len(nodes)
    if len(edges.shape)==1:
        edges = [(int(iii),int(jjj)) for iii,jjj in [edges]]
    else:
        edges = [(int(iii),int(jjj)) for iii,jjj in edges]

    # define graph
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    

   
    
    # average degree (AD)
    ave_deg_net = np.sum(A)/A.shape[0]
    

    
    # variance of degree distribution (VD)
    var_deg_net = np.sqrt(np.sum(np.square(np.sum(A,axis = 0)-ave_deg_net))/(A.shape[0]-1))
    

    # average (local) clustering coefficient (ACC)
    ave_clust_net = nx.average_clustering(G)


    # samples chosen - features
    edge_pairs_f_i = edge_s[:,0]
    edge_pairs_f_j = edge_s[:,1]

    # local number of triangles for i and j (LNT_i, LNT_j)  
    numtriang_nodes_obj = nx.triangles(G)
    numtriang_nodes = []
    for nn in range(len(nodes)):
        numtriang_nodes.append(numtriang_nodes_obj[nn])
     
    numtriang1_edges = []
    numtriang2_edges = []
    for ee in range(len(edge_s)):
        numtriang1_edges.append(numtriang_nodes[edge_s[ee][0]])
        numtriang2_edges.append(numtriang_nodes[edge_s[ee][1]])

         
    # Page rank values for i and j (PR_i, PR_j)
    page_rank_nodes_obj = nx.pagerank(G)
    page_rank_nodes = []
    for nn in range(len(nodes)):
        page_rank_nodes.append(page_rank_nodes_obj[nn])
        
    page_rank1_edges = []
    page_rank2_edges = []
    for ee in range(len(edge_s)):
        page_rank1_edges.append(page_rank_nodes[edge_s[ee][0]])
        page_rank2_edges.append(page_rank_nodes[edge_s[ee][1]])

      
    # j-th entry of the personalized page rank of node i (PPR)
    page_rank_pers_nodes = []
    hot_vec = {}
    for nn in range(len(nodes)):
        hot_vec[nn] = 0
    for nn in range(len(nodes)):
        hot_vec_copy = hot_vec.copy()
        hot_vec_copy[nn] = 1 
        page_rank_pers_nodes.append(nx.pagerank(G,personalization=hot_vec_copy))

    page_rank_pers_edges = []
    for ee in range(len(edge_s)):
        page_rank_pers_edges.append(page_rank_pers_nodes[edge_s[ee][0]][edge_s[ee][1]])

    # local clustering coefficients for i and j (LCC_i, LCC_j)
    clust_nodes_obj = nx.clustering(G)
    clust_nodes = []
    for nn in range(len(nodes)):
        clust_nodes.append(clust_nodes_obj[nn])
    
    clust1_edges = []
    clust2_edges = []
    for ee in range(len(edge_s)):
        clust1_edges.append(clust_nodes[edge_s[ee][0]])
        clust2_edges.append(clust_nodes[edge_s[ee][1]])

     
    # average neighbor degrees for i and j (AND_i, AND_j)
    ave_neigh_deg_nodes_obj = nx.average_neighbor_degree(G)
    ave_neigh_deg_nodes = []
    for nn in range(len(nodes)):
        ave_neigh_deg_nodes.append(ave_neigh_deg_nodes_obj[nn])
    
    ave_neigh_deg1_edges = []
    ave_neigh_deg2_edges = []
    for ee in range(len(edge_s)):
        ave_neigh_deg1_edges.append(ave_neigh_deg_nodes[edge_s[ee][0]])
        ave_neigh_deg2_edges.append(ave_neigh_deg_nodes[edge_s[ee][1]])
   
    # degree centralities for i and j (DC_i, DC_j)
    deg_cent_nodes_obj = nx.degree_centrality(G)
    deg_cent_nodes = []
    for nn in range(len(nodes)):
        deg_cent_nodes.append(deg_cent_nodes_obj[nn])
     
    deg_cent1_edges = []
    deg_cent2_edges = []
    for ee in range(len(edge_s)):
        deg_cent1_edges.append(deg_cent_nodes[edge_s[ee][0]])
        deg_cent2_edges.append(deg_cent_nodes[edge_s[ee][1]])


	# eigenvector centralities for i and j (EC_i, EC_j)
    tr = 1
    toler = 1e-6
    while tr == 1:
        try:
            eig_cent_nodes_obj = nx.eigenvector_centrality(G,tol = toler)
            tr = 0
        except:
            toler = toler*1e1
    
    eig_cent_nodes = []
    for nn in range(len(nodes)):
        eig_cent_nodes.append(eig_cent_nodes_obj[nn])
     
    eig_cent1_edges = []
    eig_cent2_edges = []
    for ee in range(len(edge_s)):
        eig_cent1_edges.append(eig_cent_nodes[edge_s[ee][0]])
        eig_cent2_edges.append(eig_cent_nodes[edge_s[ee][1]])

    # Katz centralities for i and j (KC_i, KC_j)
    ktz_cent_nodes_obj = nx.katz_centrality_numpy(G)
    ktz_cent_nodes = []
    for nn in range(len(nodes)):
        ktz_cent_nodes.append(ktz_cent_nodes_obj[nn])
    
    ktz_cent1_edges = []
    ktz_cent2_edges = []
    for ee in range(len(edge_s)):
        ktz_cent1_edges.append(ktz_cent_nodes[edge_s[ee][0]])
        ktz_cent2_edges.append(ktz_cent_nodes[edge_s[ee][1]])

      
    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    jacc_coeff_obj = nx.jaccard_coefficient(G,edge_s)
    jacc_coeff_edges = []
    for uu,vv,jj in jacc_coeff_obj:
        jacc_coeff_edges.append([uu,vv,jj])   
    df_jacc_coeff = pd.DataFrame(jacc_coeff_edges, columns=['i','j','jacc_coeff'])
    df_jacc_coeff['ind'] = df_jacc_coeff.index

    # resource allocation index of i, j (RA)
    res_alloc_ind_obj = nx.resource_allocation_index(G, edge_s)
    res_alloc_ind_edges = []
    for uu,vv,jj in res_alloc_ind_obj:
        res_alloc_ind_edges.append([uu,vv,jj])
    df_res_alloc_ind = pd.DataFrame(res_alloc_ind_edges, columns=['i','j','res_alloc_ind'])    
    df_res_alloc_ind['ind'] = df_res_alloc_ind.index


  	# Adamic/Adar index of i, j (AA)
    adam_adar_obj =  nx.adamic_adar_index(G, edge_s)
    adam_adar_edges = []
    for uu,vv,jj in adam_adar_obj:
        adam_adar_edges.append([uu,vv,jj])
    df_adam_adar = pd.DataFrame(adam_adar_edges, columns=['i','j','adam_adar'])
    df_adam_adar['ind'] = df_adam_adar.index
    
    df_merge = pd.merge(df_jacc_coeff,df_res_alloc_ind, on=['ind','i','j'], sort=False)
    df_merge = pd.merge(df_merge,df_adam_adar, on=['ind','i','j'], sort=False)


    # preferential attachment (degree product) of i, j (PA)
    pref_attach_obj = nx.preferential_attachment(G, edge_s)
    pref_attach_edges = []
    for uu,vv,jj in pref_attach_obj:
        pref_attach_edges.append([uu,vv,jj])
    df_pref_attach = pd.DataFrame(pref_attach_edges, columns=['i','j','pref_attach'])
    df_pref_attach['ind'] = df_pref_attach.index

             
    # global features:
    # similarity of connections in the graph with respect to the node degree
    # degree assortativity (DA)
    deg_ass_net = nx.degree_assortativity_coefficient(G)


    # transitivity: fraction of all possible triangles present in G
    # network transitivity (clustering coefficient) (NT)
    transit_net = nx.transitivity(G)  
    # network diameter (ND)


    try:
        diam_net = nx.diameter(G)
    except:
        diam_net = np.inf
        
    ave_deg_net = [ave_deg_net for ii in range(len(edge_s))]
    var_deg_net = [var_deg_net for ii in range(len(edge_s))]
    ave_clust_net = [ave_clust_net for ii in range(len(edge_s))]
    deg_ass_net = [deg_ass_net for ii in range(len(edge_s))]
    transit_net = [transit_net for ii in range(len(edge_s))]
    diam_net = [diam_net for ii in range(len(edge_s))]
    com_ne = []
    for ee in range(len(edge_s)):
        com_ne.append(len(sorted(nx.common_neighbors(G,edge_s[ee][0],edge_s[ee][1]))))

       
    # closeness centralities for i and j (CC_i, CC_j)
    closn_cent_nodes_obj = nx.closeness_centrality(G)
    closn_cent_nodes = []
    for nn in range(len(nodes)):
        closn_cent_nodes.append(closn_cent_nodes_obj[nn])
      
    closn_cent1_edges = []
    closn_cent2_edges = []
    for ee in range(len(edge_s)):
        closn_cent1_edges.append(closn_cent_nodes[edge_s[ee][0]])
        closn_cent2_edges.append(closn_cent_nodes[edge_s[ee][1]])

       
    # shortest path between i, j (SP)        
    short_Mat_aux = nx.shortest_path_length(G)
    short_Mat={}
    for ss in range(N):
        value = next(short_Mat_aux)
        short_Mat[value[0]] = value[1]   
    short_path_edges = []
    for ee in range(len(edge_s)):
        if edge_s[ee][1] in short_Mat[edge_s[ee][0]].keys():
            short_path_edges.append(short_Mat[edge_s[ee][0]][edge_s[ee][1]])  
        else:
            short_path_edges.append(np.inf)

          
    # load centralities for i and j (LC_i, LC_j)
    load_cent_nodes_obj = nx.load_centrality(G,normalized=True)
    load_cent_nodes = []
    for nn in range(len(nodes)):
        load_cent_nodes.append(load_cent_nodes_obj[nn])
    
    load_cent1_edges = []
    load_cent2_edges = []
    for ee in range(len(edge_s)):
        load_cent1_edges.append(load_cent_nodes[edge_s[ee][0]])
        load_cent2_edges.append(load_cent_nodes[edge_s[ee][1]])


    # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
    betw_cent_nodes_obj = nx.betweenness_centrality(G,normalized=True)
    betw_cent_nodes = []
    for nn in range(len(nodes)):
        betw_cent_nodes.append(betw_cent_nodes_obj[nn])
    
    betw_cent1_edges = []
    betw_cent2_edges = []
    for ee in range(len(edge_s)):
        betw_cent1_edges.append(betw_cent_nodes[edge_s[ee][0]])
        betw_cent2_edges.append(betw_cent_nodes[edge_s[ee][1]])
    
    
       
    neigh_ = {}
    for nn in range(len(nodes)):
        neigh_[nn] = np.where(A[nn,:])[0]
    
    df_pref_attach = []
    for ee in range(len(edge_s)):
        df_pref_attach.append(len(neigh_[edge_s[ee][0]])*len(neigh_[edge_s[ee][1]]))
    
    U, sig, V = np.linalg.svd(A, full_matrices=False)
    S = np.diag(sig)
    Atilda = np.dot(U, np.dot(S, V))
    Atilda = np.array(Atilda)
    
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
    svd_edges = []
    # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
    svd_edges_dot = []
    # average of entries i and j’s neighbors in low rank approximation
    svd_edges_mean = []
    for ee in range(len(edge_s)):
        svd_edges.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]]))        
 
  
    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    f_LHN = lambda num,den: 0 if (num==0 and den==0) else float(num)/den 
    LHN_edges = [f_LHN(num,den) for num,den in zip(np.array(com_ne),np.array(df_pref_attach))]
    
    U, sig, V = np.linalg.svd(A)
    S = linalg.diagsvd(sig, A.shape[0], A.shape[1])
    S_trunc = S.copy()
    S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(A.shape[0])))]] = 0
    Atilda = np.dot(np.dot(U, S_trunc), V)
    Atilda = np.array(Atilda)

 
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # an approximation of LRA (LRA-approx)
    svd_edges_approx = []
    # an approximation of dLRA (dLRA-approx)
    svd_edges_dot_approx = []
    # an approximation of mLRA (mLRA-approx)
    svd_edges_mean_approx = []
    for ee in range(len(edge_s)):
        svd_edges_approx.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot_approx.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean_approx.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]])) 
    

    # number of nodes (N)
    num_nodes = A_orig.shape[0]
    # number of observed edges (OE)
    num_edges = int(np.sum(A)/2)
    
    # construct a dictionary of the features
    d = {'i':edge_pairs_f_i, 'j':edge_pairs_f_j, 'com_ne':com_ne, 'ave_deg_net':ave_deg_net, \
         'var_deg_net':var_deg_net, 'ave_clust_net':ave_clust_net, 'num_triangles_1':numtriang1_edges, 'num_triangles_2':numtriang2_edges, \
         'page_rank_pers_edges':page_rank_pers_edges, 'pag_rank1':page_rank1_edges, 'pag_rank2':page_rank2_edges, 'clust_coeff1':clust1_edges, 'clust_coeff2':clust2_edges, 'ave_neigh_deg1':ave_neigh_deg1_edges, 'ave_neigh_deg2':ave_neigh_deg2_edges,\
         'eig_cent1':eig_cent1_edges, 'eig_cent2':eig_cent2_edges, 'deg_cent1':deg_cent1_edges, 'deg_cent2':deg_cent2_edges, 'clos_cent1':closn_cent1_edges, 'clos_cent2':closn_cent2_edges, 'betw_cent1':betw_cent1_edges, 'betw_cent2':betw_cent2_edges, \
         'load_cent1':load_cent1_edges, 'load_cent2':load_cent2_edges, 'ktz_cent1':ktz_cent1_edges, 'ktz_cent2':ktz_cent2_edges, 'pref_attach':df_pref_attach, 'LHN':LHN_edges, 'svd_edges':svd_edges,'svd_edges_dot':svd_edges_dot,'svd_edges_mean':svd_edges_mean,\
         'svd_edges_approx':svd_edges_approx,'svd_edges_dot_approx':svd_edges_dot_approx,'svd_edges_mean_approx':svd_edges_mean_approx, 'short_path':short_path_edges, 'deg_assort':deg_ass_net, 'transit_net':transit_net, 'diam_net':diam_net, \
         'num_nodes':num_nodes, 'num_edges':num_edges}     
    
    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    df_feat = pd.merge(df_feat,df_merge, on=['ind','i','j'], sort=False)
    return df_feat


def creat_full_set(df_t,df_f):
    
    """ 
    This reads dataframes created for positive and negative class, join them with their associated label.

    Input and Parameters:
    -------
    df_t: datafram of features for true edges
    df_f: datafram of features for true non-edges

    Returns
    -------
    df_all: a data frames with columns of features and ground truth 

    Examples:
    -------
    df_all = creat_full_set(df_t,df_f)
    """
    df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]
    df_all = pd.concat(df_all)
    
    # data cleaning
    df_all.loc[df_all['short_path'] == np.inf,'short_path'] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'])
    df_all.loc[df_all['diam_net'] == np.inf,'diam_net'] = 1e6
     
    return df_all


def creat_numpy_files(dir_results, df_ho, df_tr):
    
    """ 
    This function reads dataframes created for positive and negative classes, join them with their associated label.

    Input and Parameters:
    -------
    df_tr: datafram of features/ground truth for positive and negative class for model selection
    df_ho: datafram of features/ground truth for positive and negative class for held out model performance

    Returns:
    -------
    save numpy files of X_train_i and y_train_i for 5 folds, also X_Eseen/X_Eunseen, y_Eseen/y_Eunseen in dir_results

    Example:
    -------
    creat_numpy_files(dir_results, df_ho, df_tr)
    """
    
    feature_set = ['com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net',
           'num_triangles_1', 'num_triangles_2', 'page_rank_pers_edges',
           'pag_rank1', 'pag_rank2', 'clust_coeff1', 'clust_coeff2',
           'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2',
           'deg_cent1', 'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1',
           'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2',
           'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean',
           'svd_edges_approx', 'svd_edges_dot_approx', 'svd_edges_mean_approx',
           'short_path', 'deg_assort', 'transit_net', 'diam_net',
           'jacc_coeff', 'res_alloc_ind', 'adam_adar' , 'num_nodes','num_edges']  

    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    
    
    X_train_orig = df_tr
    y_train_orig = np.array(df_tr.TP)

    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        

        X_train = train.loc[:,feature_set]
        X_test = test.loc[:,feature_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=42)
        X_train, y_train = sm.fit_sample(X_train, y_train)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,feature_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  

    # balance train set with upsampling
    sm = RandomOverSampler(random_state=42)
    X_seen, y_seen = sm.fit_sample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,feature_set]
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")
    
    
def model_selection(path_to_data, path_to_results, n_depths, n_ests):
    
    """ 
    This function runs cross validation on train set and finds the random forest model parameters which yeilds to best fmeasure.

    Input and Parameters:
    -------
    path_to_data: path to held out featute matrices 
    path_to_results: path to save model performance ast txt file
    n_depth: a list of max_depths for random forest parameter
    n_est: a list of n_estimators for random forest parameter

    Returns:
    -------
    n_depth: n_depth which yeild to maximum fmeasure
    n_est: n_est which yeild to maximum fmeasure

    Examples:
    -------
    n_depth, ne_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    """
    
    fmeasure_matrix = np.zeros((len(n_depths),len(n_ests)))
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    # load train and validation set for each fold
    X_train = {}
    y_train = {}
    X_test = {}
    y_test = {}
    for nFold in range(1,6):
        
        exec("X_train["+ str(nFold) +"] = np.load( path_to_data + '/X_trainE_cv"+ str(nFold) +".npy')")
        exec("y_train["+ str(nFold) +"] = np.load( path_to_data + '/y_trainE_cv"+ str(nFold) +".npy')")
        exec("X_test["+ str(nFold) +"] = np.load( path_to_data + '/X_testE_cv"+ str(nFold) +".npy')")
        exec("y_test["+ str(nFold) +"] = np.load( path_to_data + '/y_testE_cv"+ str(nFold) +".npy')")
    
    # run a grid search for parameter tuning 
    print("start grid search ... ")
    for n_ii, ii in enumerate(n_depths):
        for n_jj, jj in enumerate(n_ests):
        
            auc_measure = []     
            precision_total = np.zeros((5,2))
            recall_total = np.zeros((5,2))
            f_measure_total = np.zeros((5,2))
            
            for cv in range(1,6):
                
                 Xtr = X_train[cv]
                 ytr = y_train[cv]
                 Xts = X_test[cv]
                 yts = y_test[cv]
                
                 # train the model 
                 
                 dtree_model = RandomForestClassifier(max_depth=ii,n_estimators=jj).fit(Xtr, ytr)
                    
                 # predict for test test
                 dtree_predictions = dtree_model.predict(Xts)
                 dtree_proba = dtree_model.predict_proba(Xts)
                        
                 # calculate performance metrics
                 cm_dt4 = confusion_matrix(yts, dtree_predictions)
                 
                 auc_aux = roc_auc_score(yts, dtree_proba[:,1])
                 auc_measure.append(auc_aux)
                 
                 precision_aux, recall_aux, f_measure_aux, _ = precision_recall_fscore_support(yts, dtree_predictions, average=None)
                 precision_total[cv-1,:] = precision_aux
                 recall_total[cv-1,:] = recall_aux
                 f_measure_total[cv-1,:] = f_measure_aux
              
            # take average of performance metrics across folds
            mean_auc = np.mean(auc_measure)
            mean_precision = np.mean(precision_total,axis=0)
            mean_recall = np.mean(recall_total,axis=0)
            mean_f_measure = np.mean(f_measure_total,axis=0)
            
            # write the result in text file
            f = open( path_to_results + '/RF_Best_metrics.txt','w')
            f.write('mean_AUC = '+ str(mean_auc)+'\n')
            f.write('mean_precision = '+ str(mean_precision)+'\n')
            f.write('mean_recall = '+ str(mean_recall)+'\n')
            f.write('mean_f_measure = '+ str(mean_f_measure)+'\n')            
            f.close()
            
            # keep track of average fmeasure for each parameter set
            
            fmeasure_matrix[n_ii,n_jj] = mean_f_measure[0]
            
    # find the model parameters which gives the best average fmeasure on 5 fold validation sets    
    i,j = np.unravel_index(fmeasure_matrix.argmax(), fmeasure_matrix.shape)
    n_depth = n_depths[i]
    ne_est = n_ests[j]
    print("best parameters for random forest are: n_depth: "+str(n_depth)+", and n_estimators: "+str(ne_est))
    return n_depth, ne_est
        
        
def heldout_performance(path_to_data, path_to_results, n_depth, n_est):
    
    """ 
    This function trains a random forest model on seen data and performs prediction on heldout.

    Input and Parameters:
    -------
    path_to_data: path to held out featute matrices 
    path_to_results: path to save model performance ast txt file
    n_depth: max_depth for random forest parameter
    n_est: n_estimators for random forest parameter

    Returns:
    -------
    auc_measure: auc on heldout
    precision_total: precision of positive class on heldout
    recall_total: recall of positive class on heldout

    Examples:
    -------
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    """
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    f = open(path_to_results + '/RF_Best_metrics.txt','w')
    #path_to_data = './feature_metrices'
    
    # read data
    X_train = np.load(path_to_data+'/X_Eseen.npy')
    y_train = np.load(path_to_data+'/y_Eseen.npy')
    X_test = np.load(path_to_data+'/X_Eunseen.npy')
    y_test = np.load(path_to_data+'/y_Eunseen.npy')
    
    
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    
    col_mean = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_mean, inds[1])
     
       
    # train the model
    dtree_model = RandomForestClassifier(n_estimators=n_est,max_depth=n_depth).fit(X_train, y_train)
    
    
    # feature importance and prediction on test set 
    feature_importance = dtree_model.feature_importances_
    dtree_predictions = dtree_model.predict(X_test)
    dtree_proba = dtree_model.predict_proba(X_test)
      
    # calculate performance metrics
    cm_dt4 = confusion_matrix(y_test, dtree_predictions)
    auc_measure = roc_auc_score(y_test, dtree_proba[:,1])
    auprc = average_precision_score(y_test, dtree_proba[:,1])
     
    precision_total, recall_total, f_measure_total, _ = precision_recall_fscore_support(y_test, dtree_predictions, average=None)
       
    
    
    f.write('heldout_AUC = '+ str(auc_measure)+'\n')
    f.write('heldout_precision = '+ str(precision_total)+'\n')
    f.write('heldout_recall = '+ str(recall_total)+'\n')
    f.write('heldout_f_measure = '+ str(f_measure_total)+'\n')
    f.write('feature_importance = '+ str(list(feature_importance))+'\n')
    f.close()
    
    print("AUC: " +str(np.round(auc_measure,2)))
    print("AUPRC: " +str(np.round(auprc,2)))
    print("precision: " +str(np.round(precision_total[0],2)))
    print("recall: " +str(np.round(recall_total[0],2)))
    print("got here")
    print(auc_measure, auprc, precision_total[0], recall_total[0])
    return auprc, auc_measure, precision_total[0], recall_total[0]


def demo(): 
    
    """ 
    This function extracts topological features and performs link prediction using stacking model on a sample network.

    Input and Parameters:
    -------

    Returns:
    -------

    Examples:
    -------
    >>> demo()
    """
    
    #### load the original netowrk A_orig
    path_E_orig = "./edge_orig.txt"
    edges_orig = np.loadtxt(path_E_orig,comments = '#')
    edges_orig = np.array(np.matrix(edges_orig))
    num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()

    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    
    #### extract features #####
    sample_true_false_edges(A_orig, A_tr, A_ho)
    edge_t_tr = np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int')
    df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
    df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)
    
    edge_t_ho = np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    df_f_ho = gen_topol_feats(A_orig, A_ho, edge_f_ho)
    df_t_ho = gen_topol_feats(A_orig, A_ho, edge_t_ho)
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auprc, auc, precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    return auprc, auc, precision, recall

def topol_stacking(edges_orig): 
    
    """ 
    This function extracts topological features and performs link prediction using stacking model on the input network (edges_orig).

    Input and Parameters:
    -------
    edges_orig: the original edge list

    Returns:
    -------

    Examples:
    -------
    >>> topol_stacking(edges_orig)
    """
    
    #### load the original netowrk A_orig
    num_nodes = int(np.max(edges_orig)) + 1
    row = np.array(edges_orig)[:,0]
    col = np.array(edges_orig)[:,1]

    data_aux = np.ones(len(row))
    A_orig = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A_orig = sparse.triu(A_orig,1) + sparse.triu(A_orig,1).transpose()
    A_orig[A_orig>0] = 1 
    A_orig = A_orig.todense()

    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_ho, A_tr = gen_tr_ho_networks(A_orig, alpha, alpha_)
    
    #### extract features #####
    sample_true_false_edges(A_orig, A_tr, A_ho)
    #sample_true_false_edges_edgebetc_1(A_orig, A_tr, A_ho)
    edge_t_tr = np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int')
    

    df_f_tr = gen_topol_feats(A_orig, A_tr, edge_f_tr)
    df_t_tr = gen_topol_feats(A_orig, A_tr, edge_t_tr)
    
    edge_t_ho = np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    df_f_ho = gen_topol_feats(A_orig, A_ho, edge_f_ho)
    df_t_ho = gen_topol_feats(A_orig, A_ho, edge_t_ho)
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auprc, auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    return auprc, auc, precision, recall

def topol_stacking_temporal(edges_orig, target_layer): 
    
    """ 
    This function extracts topological features and performs link prediction 
    using stacking model on the input layers (edges_orig).

    Input and Parameters:
    -------
    edges_orig: the original edge lists of multiple(single) layer(s)
    
    target_layer: the target layer that needs to be predicted
    
    Returns:auc of the prediction for the target_layer
    -------

    Examples:
    -------
    >>> topol_stacking(edges_orig)
    """
    #### varialbes in the loop

    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
        
    edge_t_ho = []
    edge_f_ho = []
    df_f_ho = []
    df_t_ho = []   
    
    #### load target layer A
    num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    
    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_hold, A_train = gen_tr_ho_networks(A, alpha, alpha_)
    
    for i in range(len(edges_orig)): 
        
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
        #### extract features #####
        sample_true_false_edges(A, A_tr[i], A_hold)
        edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int'))
        edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int'))
        df_f_tr.append(gen_topol_feats(A, A_tr[i], edge_f_tr[i]))
        df_t_tr.append(gen_topol_feats(A, A_tr[i], edge_t_tr[i]))
        
        edge_t_ho.append(np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int'))
        edge_f_ho.append(np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int'))
        df_f_ho.append(gen_topol_feats(A, A_hold, edge_f_ho[i]))
        df_t_ho.append(gen_topol_feats(A, A_hold, edge_t_ho[i]))
        
    
    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns
    df_t_ho_columns = df_t_ho[0].columns
    df_f_ho_columns = df_f_ho[0].columns
    
    
    v1 = np.empty(df_t_tr[0].shape)
    v2 = np.empty(df_f_tr[0].shape)
    v3 = np.empty(df_t_ho[0].shape)
    v4 = np.empty(df_f_ho[0].shape)
    # Average out for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    for i in range(len(edges_orig)):
        v1 = np.add(v1,df_t_tr[i].values)
        v2 = np.add(v2,df_f_tr[i].values)
        v3 = np.add(v3,df_t_ho[i].values)
        v4 = np.add(v4,df_f_ho[i].values)
    
    df_t_tr_ = pd.DataFrame(data=v1/len(edges_orig), columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2/len(edges_orig), columns = df_f_tr_columns)
    df_t_ho_ = pd.DataFrame(data=v3/len(edges_orig), columns = df_t_ho_columns)
    df_f_ho_ = pd.DataFrame(data=v4/len(edges_orig), columns = df_f_ho_columns)
    
    

    # Stays the same as previous function
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t')
    df_f_tr_.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho_.to_pickle(feat_path + 'df_t')
    df_f_ho_.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr_,df_f_tr_)
    df_ho = creat_full_set(df_t_ho_,df_f_ho_)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    return auc, precision, recall

def sample_true_false_edges_temporal(A_orig, A_tr, A_ho):  
    
    """ 
    This function creates the training and holdout samples.

    Input and Parameters:
    -------
    A_orig: the kth layer that is being predicted
    A_tr: the list of x-1 length that include the k-x:k-1 layers before the kth layer
    A_ho: the information that we hide out from the kth layer, i.e. A_orig

    Returns:
    -------
    nothing except it saves the files for future use

    """
    
    nsim_id = 0
    np.random.seed(nsim_id)
    
    k = len(A_tr) # number of predictors
    
    for i in range(k):
        A_diff = A_ho - A_tr[i]
        e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
        A_ho_aux = -1*A_ho + 1
        ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
        Nsamples = 10000 # number of samples
        edge_t = [] # list of true edges (positive samples)
        edge_f = [] # list of false edges (negative samples)
        for ll in range(Nsamples):
    	    edge_t_idx_aux = np.random.randint(len(e_diff[0]))
    	    edge_f_idx_aux = np.random.randint(len(ne_ho[0]))
    	    edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
    	    edge_f.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux]))
    	
    	# store for later use
        if not os.path.isdir("./edge_tf_tr/"):
    	    os.mkdir("./edge_tf_tr/")
        np.savetxt("./edge_tf_tr/edge_t_{}.txt".format(i),edge_t,fmt='%u')
        np.savetxt("./edge_tf_tr/edge_f_{}.txt".format(i),edge_f,fmt='%u')
        
    A_diff = A_orig - A_ho
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
        edge_t_idx_aux = np.random.randint(len(e_diff[0]))
        edge_f_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
        edge_f.append((ne_orig[0][edge_f_idx_aux],ne_orig[1][edge_f_idx_aux]))
        
    # store for later use
    if not os.path.isdir("./edge_tf_ho/"):
        os.mkdir("./edge_tf_ho/")
    np.savetxt("./edge_tf_ho/edge_t.txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_ho/edge_f.txt",edge_f,fmt='%u')

def creat_full_set_temporal(df_t,df_f):
    
    """ 
    This reads dataframes created for positive and negative class, join them with their associated label.

    Input and Parameters:
    -------
    df_t: datafram of features for true edges
    df_f: datafram of features for true non-edges

    Returns
    -------
    df_all: a data frames with columns of features and ground truth 

    Examples:
    -------
    df_all = creat_full_set(df_t,df_f)
    """

    df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]
    df_all = pd.concat(df_all)
    
    # data cleaning
    df_all.loc[df_all['short_path'] == np.inf,'short_path'] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'])
    df_all.loc[df_all['diam_net'] == np.inf,'diam_net'] = 1e6
     
    return df_all



def topol_stacking_multi_temporal(edges_orig, target_layer): 
    
    """ 
    This function extracts topological features and performs link prediction using stacking model on the input network (edges_orig).

    Input and Parameters:
    -------
    edges_orig: the original edge list

    Returns:
    -------

    Examples:
    -------
    >>> topol_stacking(edges_orig)
    """
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
        
    edge_t_ho = []
    edge_f_ho = []
    df_f_ho = []
    df_t_ho = []   
    
    #### load target layer A
    num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    
    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_hold, A_true = gen_tr_ho_networks(A, alpha, alpha_)

    
    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
    #A_tr.append(A_train)
    #### extract features #####
    sample_true_false_edges_temporal(A, A_tr, A_hold)
    return A, A_true, A_tr, A_hold
    
    
    for i in range(len(A_tr)):
        edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t_{}.txt".format(i)).astype('int'))
        edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f_{}.txt".format(i)).astype('int'))
        df_f_tr.append(gen_topol_feats(A, A_tr[i], edge_f_tr[i]))
        df_t_tr.append(gen_topol_feats(A, A_tr[i], edge_t_tr[i]))
        
        edge_t_ho.append(np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int'))
        edge_f_ho.append(np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int'))
        df_f_ho.append(gen_topol_feats(A, A_hold, edge_f_ho[i]))
        df_t_ho.append(gen_topol_feats(A, A_hold, edge_t_ho[i]))
    

    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns
    df_t_ho_columns = df_t_ho[0].columns
    df_f_ho_columns = df_f_ho[0].columns
    
    
    v1 = df_t_tr[0].values
    v2 = df_f_tr[0].values
    v3 = df_t_ho[0].values
    v4 = df_f_ho[0].values
    # Stack Together for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    # Let the randomforest make selection
    
    for i in range(1,len(A_tr)):
        v1 = np.vstack((v1,df_t_tr[i].values))
        v2 = np.vstack((v2,df_f_tr[i].values))
        v3 = np.vstack((v3,df_t_ho[i].values))
        v4 = np.vstack((v4,df_f_ho[i].values))
    
#    return df_t_tr, df_f_tr, df_t_ho, df_f_ho  
    df_t_tr_ = pd.DataFrame(data=v1, columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2, columns = df_f_tr_columns)
    df_t_ho_ = pd.DataFrame(data=v3, columns = df_t_ho_columns)
    df_f_ho_ = pd.DataFrame(data=v4, columns = df_f_ho_columns)   
    
    
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t')
    df_f_tr_.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho_.to_pickle(feat_path + 'df_t')
    df_f_ho_.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr_,df_f_tr_)
    df_ho = creat_full_set(df_t_ho_,df_f_ho_)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc,precision,recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)


    return auc, precision, recall

def topol_stacking_temporal_2(edges_orig, target_layer): 
    
    """ 
    This one gets the ij pair and than construct two features that average out over the 
    different layers that the contain the ij pair and does not contain the ij pair 
    It also adds on the label of the number of layer that is before the predicted layer
    
    Input and Parameters:
    -------
    edges_orig: the original edge list

    Returns:
    -------

    Examples:
    -------
    >>> topol_stacking(edges_orig)
    """
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
        
    edge_t_ho = []
    edge_f_ho = []
    df_f_ho = []
    df_t_ho = []   
    
    #### load target layer A
    num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    
    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_hold, A_train = gen_tr_ho_networks(A, alpha, alpha_)

    
    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
    #A_tr.append(A_train)
    #### extract features #####
    edge_t_bool, edge_f_bool = sample_true_false_edges_revised(A, A_train,A_tr, A_hold)
    

    edge_t_tr = np.loadtxt("./edge_tf_tr/edge_t.txt").astype('int')
    edge_f_tr = np.loadtxt("./edge_tf_tr/edge_f.txt").astype('int')
    
    df_f_tr = gen_topol_feats_temporal(A, A_tr, edge_f_tr, edge_f_bool)
    df_t_tr = gen_topol_feats_temporal(A, A_tr, edge_t_tr, edge_t_bool)

    edge_t_ho = np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho = np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    
    df_f_ho = gen_topol_feats(A, A_hold, edge_f_ho)
    df_t_ho = gen_topol_feats(A, A_hold, edge_t_ho)
    

    new_df_f_ho = pd.DataFrame(np.repeat(df_f_ho.values,2,axis=0))
    new_df_f_ho.columns = df_f_ho.columns
    
    new_df_t_ho = pd.DataFrame(np.repeat(df_t_ho.values,2,axis=0))
    new_df_t_ho.columns = df_t_ho.columns
    
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr.to_pickle(feat_path + 'df_t')
    df_f_tr.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set(df_t_tr,df_f_tr)
    df_ho = creat_full_set(df_t_ho,df_f_ho)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files(dir_output, df_ho, df_tr)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc,precision,recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)


    return auc, precision, recall


def gen_topol_feats_temporal(A_orig, A, edge_s, edge_dict): 
    
    """ 
    This function generate topoligcal feature based on which layer it is
    It also averages out over the layers that has this edge
    and the layers that does not have this edge
    
    It also takes in a list of dictionary, and than it generate a df_feat 
    that is twice as long as the original ones. For example origianlly we have
    3 10000x45 dictionaries, now we just have one 20000 dictionary, but it correponds to 
    the 10000 sample edges that is generated.
    
    
    """
    df_feat = []
    
    for i in range(len(A)):
        temp = gen_topol_feats(A_orig, A[i],edge_s)
        #sLength = len(temp['com_ne'])
        #temp['layer_ahead'] = pd.Series(np.ones(sLength)*(len(A)-i), index=temp.index)
        df_feat.append(temp)
        
    pos_feat = []
    neg_feat = []
    new_feat = []
    for x in range(len(edge_s)):
        temp1=[]
        temp2=[]
        for j in range(len(A)):
            if edge_dict[j][x] ==1:
                temp1.append(df_feat[j].iloc[x,:])
            if edge_dict[j][x] ==0:
                temp2.append(df_feat[j].iloc[x,:])
        
        if len(temp1)==0:
    
            v2=temp2[0]
            for k2 in range(len(temp2)-1):
                v2 = np.add(v2,temp2[k2])
            neg = v2/len(temp2)
            neg["existence"]= pd.Series(0, index=neg.index)
            neg_feat.append(neg)
            new_feat.append(neg)
        if len(temp2)==0:
            v1=temp1[0]
            for k1 in range(len(temp1)-1):
                v1 = np.add(v1,temp1[k1])  
            pos = v1/len(temp1)
            pos["existence"]= pd.Series(0, index=pos.index)            
            pos_feat.append(pos)
            new_feat.append(pos)
        
        if len(temp1)!=0 and len(temp2)!=0:
            v1=temp1[0]
            for k1 in range(len(temp1)-1):
                v1 = np.add(v1,temp1[k1])
    
            v2=temp2[0]
            for k2 in range(len(temp2)-1):
                v2 = np.add(v2,temp2[k2])
            
            pos = v1/len(temp1)
            pos["existence"]= pd.Series(0, index=pos.index)            
            pos_feat.append(pos)
            new_feat.append(pos)
            neg = v2/len(temp2)
            neg["existence"]= pd.Series(0, index=neg.index)
            neg_feat.append(neg)
            new_feat.append(neg)
    
    df = pd.DataFrame(new_feat, columns=df_feat[0].columns)
    
    return df
    


def sample_true_false_edges_revised(A_orig, A_true, A_train, A_ho):  
    
    """ 
    For each edge pair, it also generates a dictionary,
    the dictionary key is the layer before hands,
    and dictionary value is if this edge exist in the corredsponding layer or not
    if exists, than it is 1, if not than it is 0
    Based on the ansewr of this dictionary, we average out the df_feat that we get
    from gen_topol_features
    Note that gen_topol_features should only work on A_tr, which is the subset 
    of the predicted layer, all the samples should stay the same
    After we have sampled edges, we just took these edges and than seek the average features
    of the corresponding layers
    For example, each row represent one pair of edge in the df_feat dataframe,
    we took a row and we check the corresponding list of dictionary, and than we check
    which layer it belongs to and we average over this layer
    and than we average out over another layer that does not have the corresponding edge
    so than turns the df_feat into two different rows that corresponding to the average result
    
    So we need to output a list of dictionary for this function
    
    This list should be 10000 in length, each entry corresponding to an sampled edge.
    
    """
    
    edge_t, edge_f, edge_tt, edge_ff = sample_true_false_edges(A_orig, A_true, A_ho)
    
    A_edge_t = []
    A_edge_f = []
    for i in range(len(A_train)):
        nodes, edge_tr = adj_to_nodes_edges(A_train[i])
        nsim_id = 0
        np.random.seed(nsim_id)
    
        A_diff = A_ho - A_train[i]
        e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
        
        A_ho_aux = -1*A_ho + 1
        ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
        Nsamples =  len(e_diff[0])# number of edges
        temp_edge_t = [] # list of true edges (positive samples)
        temp_edge_f = [] # list of false edges (negative samples)
        for ll in range(Nsamples):
    	    temp_edge_t.append((e_diff[0][ll],e_diff[1][ll]))
    	    temp_edge_f.append((ne_ho[0][ll],ne_ho[1][ll]))
        A_edge_t.append(temp_edge_t)
        A_edge_f.append(temp_edge_f)
    
    edge_t_bool = []
    edge_f_bool = []
    for j in range(len(A_edge_t)):
        edge_t_bool.append([ x in A_edge_t[j] for x in edge_t])
        edge_f_bool.append([ x in A_edge_f[j] for x in edge_f])
    return edge_t_bool, edge_f_bool

""


"""
10/29 revised, sampling uniformly from the previous n layer with k layers of feature.
"""

""
def sample_true_false_edges_revised_3(A_orig, A_true, A_train, A_ho,predict_num, name):  
    
    """ 
    For each edge pair,it sample not only from the last layer that we wanted to predict
    but also from previous layers that might have useful information in them i.e. basically
    if we are given 9 layers before as information, and we consider the features of 4 layers
    than we are saying that we want to sample from the 7 choices and only consider the last
    of those 4 layers.
    1,2,3,4,5,6,7,8,9,10
    10 being the layer that needs to be predicted. Take 4 layer of information
    for example: 4,5,6,7
    than we consider the label of the edge true edge if it appears in 7 and false if not
    After that we consider take the features from all 4,5,6,7 and concatenate them
    
    Previously if we sample 10000 edges than we have 10000x45 as our output feature vector
    Now we have 10000 x (45*4) i,e, there are 180 features for 1 pair of edge.
    
    
    """
    Nsamples = 10000
    nsim_id = 0
    np.random.seed(nsim_id)
    num_sample_edges = K_random_number_add_up_to_sum(len(A_train)-predict_num+1, Nsamples)
    print(num_sample_edges)
    
    A_edge_t = []
    A_edge_f = []
    for i in range(predict_num-1, len(A_train)):
        nodes, edges = adj_to_nodes_edges(A_train[i])
        
        pos_edges = sparse.find(sparse.triu(A_train[i],1)) # true candidates
        A_neg = -1*A_train[i] + 1
        neg_edges = sparse.find(sparse.triu(A_neg,1)) # false candidates

        temp_edge_t = [] # list of true edges (positive samples)
        temp_edge_f = [] # list of false edges (negative samples)
        for ll in range(num_sample_edges[i-predict_num+1]):
    	    edge_t_idx_aux = np.random.randint(len(pos_edges[0]))
    	    edge_f_idx_aux = np.random.randint(len(neg_edges[0]))
    	    temp_edge_t.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux]))
    	    temp_edge_f.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux]))
        
        A_edge_t.append(temp_edge_t)
        A_edge_f.append(temp_edge_f)
        # store for later use
        if not os.path.isdir("./edge_tf_tr/"):
    	    os.mkdir("./edge_tf_tr/")
        np.savetxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt",temp_edge_t,fmt='%u')
        np.savetxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt",temp_edge_f,fmt='%u')
        
    
    nodes, edge_tr = adj_to_nodes_edges(A_true)
    nsim_id = 0
    np.random.seed(nsim_id)

    A_diff = A_ho - A_true
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_ho_aux = -1*A_ho + 1
    ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
    Nsamples = num_sample_edges[-1] # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    for ll in range(num_sample_edges[-1]):
	    edge_t_idx_aux = np.random.randint(len(e_diff[0]))
	    edge_f_idx_aux = np.random.randint(len(ne_ho[0]))
	    edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
	    edge_f.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux]))
	
	# store for later use
    if not os.path.isdir("./edge_tf_tr/"):
	    os.mkdir("./edge_tf_tr/")
    np.savetxt("./edge_tf_tr/edge_t"+"_"+str(name)+".txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_tr/edge_f"+"_"+str(name)+".txt",edge_f,fmt='%u')
    
    A_diff = A_orig - A_ho
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_tt = [] # list of true edges (positive samples)
    edge_ff = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
        edge_tt_idx_aux = np.random.randint(len(e_diff[0]))
        edge_ff_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_tt.append((e_diff[0][edge_tt_idx_aux],e_diff[1][edge_tt_idx_aux]))
        edge_ff.append((ne_orig[0][edge_ff_idx_aux],ne_orig[1][edge_ff_idx_aux]))
    
    # store for later use
    if not os.path.isdir("./edge_tf_ho/"):
        os.mkdir("./edge_tf_ho/")
    np.savetxt("./edge_tf_ho/edge_t"+"_"+str(name)+".txt",edge_tt,fmt='%u')
    np.savetxt("./edge_tf_ho/edge_f"+"_"+str(name)+".txt",edge_ff,fmt='%u')
    


def gen_topol_feats_3(A_orig, A_tr, edge_s): 
    
    """ 
    This function generate topoligcal feature based on which layer it is
    It also averages out over the layers that has this edge
    and the layers that does not have this edge
    
    It also takes in a list of dictionary, and than it generate a df_feat 
    that is twice as long as the original ones. For example origianlly we have
    3 10000x45 dictionaries, now we just have one 20000 dictionary, but it correponds to 
    the 10000 sample edges that is generated.
    
    
    """
    time_count = []
    temp,time_cost = gen_topol_feats_timed_3_1( A_tr[0],edge_s)
    time_count.append(time_cost)
    df_feat = temp
    for i in range(1,len(A_tr)):
        temp,time_cost = gen_topol_feats_timed_3_1(A_tr[i],edge_s)
        time_count.append(time_cost)
        df_feat = pd.concat([df_feat, temp], axis=1)
        
    temp,time_cost = gen_topol_feats_timed_3_1(A_orig, edge_s)
    time_count.append(time_cost)
    df_feat = pd.concat([df_feat, temp], axis=1)
    
    
    return df_feat, time_count
    


def topol_stacking_temporal_3(edges_orig, target_layer, predict_num): 
    
    """ 
    len(edges_orig)  =10
    predict_num = 4
    """
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
    
    
    #### load target layer A
    num_nodes = int(max( int(np.max(target_layer)) + 1, np.amax(edges_orig)))+1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    
    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix
    A_hold, A_true = gen_tr_ho_networks(A, alpha, alpha_)

    
    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
    #A_tr.append(A_train)
    #### extract features #####
    sample_true_false_edges_revised_3(A, A_true, A_tr, A_hold, predict_num, name)
    
    A_tr1 = A_tr.copy()
    A_tr.append(A_hold)
    A_tr1.append(A_true)
    
    mytime = []
    
    
    for i in range(predict_num-1, len(A_tr)-1):
        edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t_{}.txt".format(i)).astype('int'))
        edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f_{}.txt".format(i)).astype('int'))
        df_temp, time_temp = gen_topol_feats_3(A, A_tr[i-predict_num+1 :i+1], edge_f_tr[i-predict_num+1])
        df_f_tr.append(df_temp)
        mytime.append(time_temp)
      
        df_temp, time_temp = gen_topol_feats_3(A, A_tr[i-predict_num+1 :i+1], edge_t_tr[i-predict_num+1])
        df_t_tr.append(df_temp)
        mytime.append(time_temp)
        
    
#    edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t.txt".format(i)).astype('int'))
#    edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f.txt".format(i)).astype('int'))
#    df_temp, time_temp = gen_topol_feats_3(A, A_tr1[-predict_num:], edge_t_tr[-1])
#    df_t_tr.append(df_temp)
#    mytime.append(time_temp)
#    df_temp, time_temp = gen_topol_feats_3(A, A_tr1[-predict_num:], edge_f_tr[-1])
#    df_f_tr.append(df_temp)
#    mytime.append(time_temp) 

    
    
    edge_t_ho= np.loadtxt("./edge_tf_ho/edge_t.txt").astype('int')
    edge_f_ho= np.loadtxt("./edge_tf_ho/edge_f.txt").astype('int')
    df_f_ho, time1 = gen_topol_feats_3(A, A_tr[ -predict_num:], edge_f_ho)
    df_t_ho, time2 = gen_topol_feats_3(A, A_tr[ -predict_num:], edge_t_ho)
    
    mytime.append(time1)
    mytime.append(time2)
    

#    return df_f_tr, df_t_tr, df_f_ho
    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns    
    
    v1 = df_t_tr[0].values
    v2 = df_f_tr[0].values

    
    # Stack Together for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    # Let the randomforest make selection
    
    for i in range(len(A_tr)-predict_num):
        v1 = np.vstack((v1,df_t_tr[i].values))
        v2 = np.vstack((v2,df_f_tr[i].values))
    
#    return df_t_tr, df_f_tr, df_t_ho, df_f_ho  
    df_t_tr_ = pd.DataFrame(data=v1, columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2, columns = df_f_tr_columns)
    
    
    column_name = list(df_t_tr_.columns)
    
    for j in range(1,predict_num):  
        for i in range (44*j, 44*(j+1)):
            column_name[i] = column_name[i]+"_"+str(j)
    
    df_t_tr_.columns = column_name
    df_f_tr_.columns = column_name
    df_t_ho.columns = column_name
    df_f_ho.columns = column_name
    
    feat_path = "./ef_gen_tr/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t')
    df_f_tr_.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set_3(df_t_tr_,df_f_tr_,predict_num)
    df_ho = creat_full_set_3(df_t_ho,df_f_ho,predict_num)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'  # output path
    creat_numpy_files_3(dir_output, df_ho, df_tr,predict_num)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' 
    path_to_results = './results'
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auc,precision,recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)


    return auc, precision, recall
#, mytime

def K_random_number_add_up_to_sum(k,summation):
    """
    The idea of this function is to generate 7 numbers that adds up to 10000
    so that we know how much we should sample from each of the layers randomly
    Check here for the details of the method of the algorithm:
    https://math.stackexchange.com/questions/1276206/method-of-generating-random-numbers-that-sum-to-100-is-this-truly-random
    """
    x = random.sample(range(summation),k-1)
    x.sort()
    output= []
    output.append(x[0]-1)
    for i in range(1,k-1):
        output.append(x[i]-x[i-1]-1)
   
    output.append(summation-x[-1])
    random.shuffle(output)
    return output
    
def creat_full_set_3(df_t,df_f,predict_num):
    
    """ 
    This reads dataframes created for positive and negative class, join them with their associated label.

    Input and Parameters:
    -------
    df_t: datafram of features for true edges
    df_f: datafram of features for true non-edges

    Returns
    -------
    df_all: a data frames with columns of features and ground truth 

    Examples:
    -------
    df_all = creat_full_set(df_t,df_f)
    """

    df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]
    df_all = pd.concat(df_all)
    
    df_all.loc[df_all['short_path'] == np.inf,'short_path'] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'],default=0)
    df_all.loc[df_all['diam_net'] == np.inf,'diam_net'] = 1e6
    
    # data cleaning
    for i in range(1,predict_num+1):
        df_all.loc[df_all['short_path_{}'.format(i)] == np.inf,'short_path_{}'.format(i)] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'],default=0)
        df_all.loc[df_all['diam_net_{}'.format(i)] == np.inf,'diam_net_{}'.format(i)] = 1e6
     
    return df_all

    
def creat_numpy_files_3(dir_results, df_ho, df_tr, predict_num):
    
    """ 
    This functiion multiplied the column number and attache verything that is important in the function
    """
    
    feature_set = ['com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net',
           'num_triangles_1', 'num_triangles_2', 
           'pag_rank1', 'pag_rank2', 'clust_coeff1', 'clust_coeff2',
           'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2',
           'deg_cent1', 'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1',
           'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2',
           'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean',
           'svd_edges_approx', 'svd_edges_dot_approx', 'svd_edges_mean_approx',
           'short_path', 'deg_assort', 'transit_net', 'diam_net',
           'jacc_coeff', 'res_alloc_ind', 'adam_adar' , 'num_nodes','num_edges']  
    #'page_rank_pers_edges', removed
    
    
    full_feat_set = [None] * ((predict_num)*len(feature_set))
    len_of_feat = len(feature_set)
    
    for k in range(len_of_feat):
        full_feat_set[k] = feature_set[k]
    
    for j in range(1,predict_num):  
        for i in range (len_of_feat*j, len_of_feat*(j+1)):
            full_feat_set[i] = feature_set[i-len_of_feat*j]+"_"+str(j)
        
    #print(full_feat_set)
    
    
    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    
    
    X_train_orig = df_tr
    y_train_orig = np.array(df_tr.TP)

    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        

        X_train = train.loc[:,full_feat_set]
        X_test = test.loc[:,full_feat_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=len_of_feat)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,full_feat_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  

    # balance train set with upsampling
    sm = RandomOverSampler(random_state=len_of_feat)
    X_seen, y_seen = sm.fit_resample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,full_feat_set]
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")
    
def gen_topol_feats_timed_3( A, edge_s): 
    
    """ 
    This function generates the topological features for matrix A (A_tr or A_ho) over edge samples edge_s (edge_tr or edge_ho).

    Input and Parameters:
    -------
    A: the training or holdout adjacency matrix that the topological features are going to be computed over
    A_orig: the original adjacency matrix
    edge_s: the sample set of training or holdout edges that the topological features are going to be computed over

    Returns:
    -------
    df_feat: data frame of features

    Examples:
    -------
    >>> gen_topol_feats(A_orig, A_tr, edge_tr)
    >>> gen_topol_feats(A_orig, A_ho, edge_ho)
    """
    
    time_cost = {}
    _, edges = adj_to_nodes_edges(A)    
    nodes = [int(iii) for iii in range(A.shape[0])]
    N = len(nodes)
    if len(edges.shape)==1:
        edges = [(int(iii),int(jjj)) for iii,jjj in [edges]]
    else:
        edges = [(int(iii),int(jjj)) for iii,jjj in edges]

    # define graph
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    

    start_time = time.time()    
    
    # average degree (AD)
    ave_deg_net = np.sum(A)/A.shape[0]
    
    time_cost["AD"] = (time.time() - start_time)
    start_time = time.time()
    
    # variance of degree distribution (VD)
    var_deg_net = np.sqrt(np.sum(np.square(np.sum(A,axis = 0)-ave_deg_net))/(A.shape[0]-1))
    
    time_cost["VD"] = (time.time() - start_time)
    start_time = time.time()    
    
    # average (local) clustering coefficient (ACC)
    ave_clust_net = nx.average_clustering(G)

    time_cost["ACC"] = (time.time() - start_time)
    start_time = time.time()

    
    # samples chosen - features
    edge_pairs_f_i = edge_s[:,0]
    edge_pairs_f_j = edge_s[:,1]

    time_cost["chosen features"] = (time.time() - start_time)
    start_time = time.time()    

    # local number of triangles for i and j (LNT_i, LNT_j)  
    numtriang_nodes_obj = nx.triangles(G)
    numtriang_nodes = []
    for nn in range(len(nodes)):
        numtriang_nodes.append(numtriang_nodes_obj[nn])
     
    numtriang1_edges = []
    numtriang2_edges = []
    for ee in range(len(edge_s)):
        numtriang1_edges.append(numtriang_nodes[edge_s[ee][0]])
        numtriang2_edges.append(numtriang_nodes[edge_s[ee][1]])

    time_cost["LNT"] = (time.time() - start_time)
    start_time = time.time()

         
    # Page rank values for i and j (PR_i, PR_j)
    page_rank_nodes_obj = nx.pagerank(G)
    page_rank_nodes = []
    for nn in range(len(nodes)):
        page_rank_nodes.append(page_rank_nodes_obj[nn])
        
    page_rank1_edges = []
    page_rank2_edges = []
    for ee in range(len(edge_s)):
        page_rank1_edges.append(page_rank_nodes[edge_s[ee][0]])
        page_rank2_edges.append(page_rank_nodes[edge_s[ee][1]])

    time_cost["Page rank"] = (time.time() - start_time)
    start_time = time.time()
        
    # j-th entry of the personalized page rank of node i (PPR)
    page_rank_pers_nodes = []
    hot_vec = {}
    for nn in range(len(nodes)):
        hot_vec[nn] = 0
    for nn in range(len(nodes)):
        hot_vec_copy = hot_vec.copy()
        hot_vec_copy[nn] = 1 
        page_rank_pers_nodes.append(nx.pagerank(G,personalization=hot_vec_copy))

    page_rank_pers_edges = []
    for ee in range(len(edge_s)):
        page_rank_pers_edges.append(page_rank_pers_nodes[edge_s[ee][0]][edge_s[ee][1]])

    time_cost["PPR"] = (time.time() - start_time)
    start_time = time.time()


    # local clustering coefficients for i and j (LCC_i, LCC_j)
    clust_nodes_obj = nx.clustering(G)
    clust_nodes = []
    for nn in range(len(nodes)):
        clust_nodes.append(clust_nodes_obj[nn])
    
    clust1_edges = []
    clust2_edges = []
    for ee in range(len(edge_s)):
        clust1_edges.append(clust_nodes[edge_s[ee][0]])
        clust2_edges.append(clust_nodes[edge_s[ee][1]])

    time_cost["LCC_i"] = (time.time() - start_time)
    start_time = time.time()
        
    # average neighbor degrees for i and j (AND_i, AND_j)
    ave_neigh_deg_nodes_obj = nx.average_neighbor_degree(G)
    ave_neigh_deg_nodes = []
    for nn in range(len(nodes)):
        ave_neigh_deg_nodes.append(ave_neigh_deg_nodes_obj[nn])
    
    ave_neigh_deg1_edges = []
    ave_neigh_deg2_edges = []
    for ee in range(len(edge_s)):
        ave_neigh_deg1_edges.append(ave_neigh_deg_nodes[edge_s[ee][0]])
        ave_neigh_deg2_edges.append(ave_neigh_deg_nodes[edge_s[ee][1]])
 
    time_cost["AND_i"] = (time.time() - start_time)
    start_time = time.time()
    
    # degree centralities for i and j (DC_i, DC_j)
    deg_cent_nodes_obj = nx.degree_centrality(G)
    deg_cent_nodes = []
    for nn in range(len(nodes)):
        deg_cent_nodes.append(deg_cent_nodes_obj[nn])
     
    deg_cent1_edges = []
    deg_cent2_edges = []
    for ee in range(len(edge_s)):
        deg_cent1_edges.append(deg_cent_nodes[edge_s[ee][0]])
        deg_cent2_edges.append(deg_cent_nodes[edge_s[ee][1]])

    time_cost["DC_i"] = (time.time() - start_time)
    start_time = time.time()

	# eigenvector centralities for i and j (EC_i, EC_j)
    tr = 1
    toler = 1e-6
    while tr == 1:
        try:
            eig_cent_nodes_obj = nx.eigenvector_centrality(G,tol = toler)
            tr = 0
        except:
            toler = toler*1e1
    
    eig_cent_nodes = []
    for nn in range(len(nodes)):
        eig_cent_nodes.append(eig_cent_nodes_obj[nn])
     
    eig_cent1_edges = []
    eig_cent2_edges = []
    for ee in range(len(edge_s)):
        eig_cent1_edges.append(eig_cent_nodes[edge_s[ee][0]])
        eig_cent2_edges.append(eig_cent_nodes[edge_s[ee][1]])

    time_cost["EC_i"] = (time.time() - start_time)
    start_time = time.time()

    # Katz centralities for i and j (KC_i, KC_j)
    ktz_cent_nodes_obj = nx.katz_centrality_numpy(G)
    ktz_cent_nodes = []
    for nn in range(len(nodes)):
        ktz_cent_nodes.append(ktz_cent_nodes_obj[nn])
    
    ktz_cent1_edges = []
    ktz_cent2_edges = []
    for ee in range(len(edge_s)):
        ktz_cent1_edges.append(ktz_cent_nodes[edge_s[ee][0]])
        ktz_cent2_edges.append(ktz_cent_nodes[edge_s[ee][1]])

    time_cost["KC_i"] = (time.time() - start_time)
    start_time = time.time()
        
    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    jacc_coeff_obj = nx.jaccard_coefficient(G,edge_s)
    jacc_coeff_edges = []
    for uu,vv,jj in jacc_coeff_obj:
        jacc_coeff_edges.append([uu,vv,jj])   
    df_jacc_coeff = pd.DataFrame(jacc_coeff_edges, columns=['i','j','jacc_coeff'])
    df_jacc_coeff['ind'] = df_jacc_coeff.index

    time_cost["JC"] = (time.time() - start_time)
    start_time = time.time()


    # resource allocation index of i, j (RA)
    res_alloc_ind_obj = nx.resource_allocation_index(G, edge_s)
    res_alloc_ind_edges = []
    for uu,vv,jj in res_alloc_ind_obj:
        res_alloc_ind_edges.append([uu,vv,jj])
    df_res_alloc_ind = pd.DataFrame(res_alloc_ind_edges, columns=['i','j','res_alloc_ind'])    
    df_res_alloc_ind['ind'] = df_res_alloc_ind.index

    time_cost["RA"] = (time.time() - start_time)
    start_time = time.time()
  
  	# Adamic/Adar index of i, j (AA)
    adam_adar_obj =  nx.adamic_adar_index(G, edge_s)
    adam_adar_edges = []
    for uu,vv,jj in adam_adar_obj:
        adam_adar_edges.append([uu,vv,jj])
    df_adam_adar = pd.DataFrame(adam_adar_edges, columns=['i','j','adam_adar'])
    df_adam_adar['ind'] = df_adam_adar.index
    
    df_merge = pd.merge(df_jacc_coeff,df_res_alloc_ind, on=['ind','i','j'], sort=False)
    df_merge = pd.merge(df_merge,df_adam_adar, on=['ind','i','j'], sort=False)

    time_cost["AA"] = (time.time() - start_time)
    start_time = time.time()

    # preferential attachment (degree product) of i, j (PA)
    pref_attach_obj = nx.preferential_attachment(G, edge_s)
    pref_attach_edges = []
    for uu,vv,jj in pref_attach_obj:
        pref_attach_edges.append([uu,vv,jj])
    df_pref_attach = pd.DataFrame(pref_attach_edges, columns=['i','j','pref_attach'])
    df_pref_attach['ind'] = df_pref_attach.index

    time_cost["PA"] = (time.time() - start_time)
    start_time = time.time()
                
    # global features:
    # similarity of connections in the graph with respect to the node degree
    # degree assortativity (DA)
    deg_ass_net = nx.degree_assortativity_coefficient(G)

    time_cost["DA"] = (time.time() - start_time)
    start_time = time.time()

    # transitivity: fraction of all possible triangles present in G
    # network transitivity (clustering coefficient) (NT)
    transit_net = nx.transitivity(G)  
    # network diameter (ND)

    time_cost["NT"] = (time.time() - start_time)
    start_time = time.time()

    try:
        diam_net = nx.diameter(G)
    except:
        diam_net = np.inf
        
    ave_deg_net = [ave_deg_net for ii in range(len(edge_s))]
    var_deg_net = [var_deg_net for ii in range(len(edge_s))]
    ave_clust_net = [ave_clust_net for ii in range(len(edge_s))]
    deg_ass_net = [deg_ass_net for ii in range(len(edge_s))]
    transit_net = [transit_net for ii in range(len(edge_s))]
    diam_net = [diam_net for ii in range(len(edge_s))]
    com_ne = []
    for ee in range(len(edge_s)):
        com_ne.append(len(sorted(nx.common_neighbors(G,edge_s[ee][0],edge_s[ee][1]))))

    time_cost["global_stuff"] = (time.time() - start_time)
    start_time = time.time()
         
    # closeness centralities for i and j (CC_i, CC_j)
    closn_cent_nodes_obj = nx.closeness_centrality(G)
    closn_cent_nodes = []
    for nn in range(len(nodes)):
        closn_cent_nodes.append(closn_cent_nodes_obj[nn])
      
    closn_cent1_edges = []
    closn_cent2_edges = []
    for ee in range(len(edge_s)):
        closn_cent1_edges.append(closn_cent_nodes[edge_s[ee][0]])
        closn_cent2_edges.append(closn_cent_nodes[edge_s[ee][1]])

    time_cost["CC_i"] = (time.time() - start_time)
    start_time = time.time()
          
    # shortest path between i, j (SP)        
    short_Mat_aux = nx.shortest_path_length(G)
    short_Mat={}
    for ss in range(N):
        value = next(short_Mat_aux)
        short_Mat[value[0]] = value[1]   
    short_path_edges = []
    for ee in range(len(edge_s)):
        if edge_s[ee][1] in short_Mat[edge_s[ee][0]].keys():
            short_path_edges.append(short_Mat[edge_s[ee][0]][edge_s[ee][1]])  
        else:
            short_path_edges.append(np.inf)

    time_cost["SP"] = (time.time() - start_time)
    start_time = time.time()
            
    # load centralities for i and j (LC_i, LC_j)
    load_cent_nodes_obj = nx.load_centrality(G,normalized=True)
    load_cent_nodes = []
    for nn in range(len(nodes)):
        load_cent_nodes.append(load_cent_nodes_obj[nn])
    
    load_cent1_edges = []
    load_cent2_edges = []
    for ee in range(len(edge_s)):
        load_cent1_edges.append(load_cent_nodes[edge_s[ee][0]])
        load_cent2_edges.append(load_cent_nodes[edge_s[ee][1]])

    time_cost["LC_i"] = (time.time() - start_time)
    start_time = time.time()


    # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
    betw_cent_nodes_obj = nx.betweenness_centrality(G,normalized=True)
    betw_cent_nodes = []
    for nn in range(len(nodes)):
        betw_cent_nodes.append(betw_cent_nodes_obj[nn])
    
    betw_cent1_edges = []
    betw_cent2_edges = []
    for ee in range(len(edge_s)):
        betw_cent1_edges.append(betw_cent_nodes[edge_s[ee][0]])
        betw_cent2_edges.append(betw_cent_nodes[edge_s[ee][1]])
    
    
    time_cost["SPBC_i"] = (time.time() - start_time)
    start_time = time.time()   
        
    neigh_ = {}
    for nn in range(len(nodes)):
        neigh_[nn] = np.where(A[nn,:])[0]
    
    df_pref_attach = []
    for ee in range(len(edge_s)):
        df_pref_attach.append(len(neigh_[edge_s[ee][0]])*len(neigh_[edge_s[ee][1]]))
    
    U, sig, V = np.linalg.svd(A, full_matrices=False)
    S = np.diag(sig)
    Atilda = np.dot(U, np.dot(S, V))
    Atilda = np.array(Atilda)
    
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
    svd_edges = []
    # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
    svd_edges_dot = []
    # average of entries i and j’s neighbors in low rank approximation
    svd_edges_mean = []
    for ee in range(len(edge_s)):
        svd_edges.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]]))        
 
    time_cost["SVD"] = (time.time() - start_time)
    start_time = time.time()
    
    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    f_LHN = lambda num,den: 0 if (num==0 and den==0) else float(num)/den 
    LHN_edges = [f_LHN(num,den) for num,den in zip(np.array(com_ne),np.array(df_pref_attach))]
    
    U, sig, V = np.linalg.svd(A)
    S = linalg.diagsvd(sig, A.shape[0], A.shape[1])
    S_trunc = S.copy()
    S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(A.shape[0])))]] = 0
    Atilda = np.dot(np.dot(U, S_trunc), V)
    Atilda = np.array(Atilda)

    time_cost["LHN"] = (time.time() - start_time)
    start_time = time.time()
   
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # an approximation of LRA (LRA-approx)
    svd_edges_approx = []
    # an approximation of dLRA (dLRA-approx)
    svd_edges_dot_approx = []
    # an approximation of mLRA (mLRA-approx)
    svd_edges_mean_approx = []
    for ee in range(len(edge_s)):
        svd_edges_approx.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot_approx.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean_approx.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]])) 
    
    time_cost["dLRA"] = (time.time() - start_time)
    start_time = time.time()

    # number of nodes (N)
    num_nodes = A.shape[0]
    # number of observed edges (OE)
    num_edges = int(np.sum(A)/2)
    
    # construct a dictionary of the features
    d = {'i':edge_pairs_f_i, 'j':edge_pairs_f_j, 'com_ne':com_ne, 'ave_deg_net':ave_deg_net, \
         'var_deg_net':var_deg_net, 'ave_clust_net':ave_clust_net, 'num_triangles_1':numtriang1_edges, 'num_triangles_2':numtriang2_edges, \
         'page_rank_pers_edges':page_rank_pers_edges, 'pag_rank1':page_rank1_edges, 'pag_rank2':page_rank2_edges, 'clust_coeff1':clust1_edges, 'clust_coeff2':clust2_edges, 'ave_neigh_deg1':ave_neigh_deg1_edges, 'ave_neigh_deg2':ave_neigh_deg2_edges,\
         'eig_cent1':eig_cent1_edges, 'eig_cent2':eig_cent2_edges, 'deg_cent1':deg_cent1_edges, 'deg_cent2':deg_cent2_edges, 'clos_cent1':closn_cent1_edges, 'clos_cent2':closn_cent2_edges, 'betw_cent1':betw_cent1_edges, 'betw_cent2':betw_cent2_edges, \
         'load_cent1':load_cent1_edges, 'load_cent2':load_cent2_edges, 'ktz_cent1':ktz_cent1_edges, 'ktz_cent2':ktz_cent2_edges, 'pref_attach':df_pref_attach, 'LHN':LHN_edges, 'svd_edges':svd_edges,'svd_edges_dot':svd_edges_dot,'svd_edges_mean':svd_edges_mean,\
         'svd_edges_approx':svd_edges_approx,'svd_edges_dot_approx':svd_edges_dot_approx,'svd_edges_mean_approx':svd_edges_mean_approx, 'short_path':short_path_edges, 'deg_assort':deg_ass_net, 'transit_net':transit_net, 'diam_net':diam_net, \
         'num_nodes':num_nodes, 'num_edges':num_edges}     
    
    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    df_feat = pd.merge(df_feat,df_merge, on=['ind','i','j'], sort=False)
    return df_feat, time_cost

def gen_topol_feats_timed_3_1(A, edge_s): 
    
    """ 
    REMOVED PPR
    """
    
    time_cost = {}
    _, edges = adj_to_nodes_edges(A)    
    nodes = [int(iii) for iii in range(A.shape[0])]
    N = len(nodes)
    if len(edges.shape)==1:
        edges = [(int(iii),int(jjj)) for iii,jjj in [edges]]
    else:
        edges = [(int(iii),int(jjj)) for iii,jjj in edges]

    # define graph
    G=nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    

    start_time = time.time()    
    
    # average degree (AD)
    ave_deg_net = np.sum(A)/A.shape[0]
    
    time_cost["AD"] = (time.time() - start_time)
    start_time = time.time()
    
    # variance of degree distribution (VD)
    var_deg_net = np.sqrt(np.sum(np.square(np.sum(A,axis = 0)-ave_deg_net))/(A.shape[0]-1))
    
    time_cost["VD"] = (time.time() - start_time)
    start_time = time.time()    
    
    # average (local) clustering coefficient (ACC)
    ave_clust_net = nx.average_clustering(G)

    time_cost["ACC"] = (time.time() - start_time)
    start_time = time.time()

    
    # samples chosen - features
    edge_pairs_f_i = edge_s[:,0]
    edge_pairs_f_j = edge_s[:,1]

    time_cost["chosen features"] = (time.time() - start_time)
    start_time = time.time()    

    # local number of triangles for i and j (LNT_i, LNT_j)  
    numtriang_nodes_obj = nx.triangles(G)
    numtriang_nodes = []
    for nn in range(len(nodes)):
        numtriang_nodes.append(numtriang_nodes_obj[nn])
     
    numtriang1_edges = []
    numtriang2_edges = []
    for ee in range(len(edge_s)):
        numtriang1_edges.append(numtriang_nodes[edge_s[ee][0]])
        numtriang2_edges.append(numtriang_nodes[edge_s[ee][1]])

    time_cost["LNT"] = (time.time() - start_time)
    start_time = time.time()

         
    # Page rank values for i and j (PR_i, PR_j)
    page_rank_nodes_obj = nx.pagerank(G)
    page_rank_nodes = []
    for nn in range(len(nodes)):
        page_rank_nodes.append(page_rank_nodes_obj[nn])
        
    page_rank1_edges = []
    page_rank2_edges = []
    for ee in range(len(edge_s)):
        page_rank1_edges.append(page_rank_nodes[edge_s[ee][0]])
        page_rank2_edges.append(page_rank_nodes[edge_s[ee][1]])

    time_cost["Page rank"] = (time.time() - start_time)
    start_time = time.time()
        
#    # j-th entry of the personalized page rank of node i (PPR)
#    page_rank_pers_nodes = []
#    hot_vec = {}
#    for nn in range(len(nodes)):
#        hot_vec[nn] = 0
#    for nn in range(len(nodes)):
#        hot_vec_copy = hot_vec.copy()
#        hot_vec_copy[nn] = 1 
#        page_rank_pers_nodes.append(nx.pagerank(G,personalization=hot_vec_copy))
#
#    page_rank_pers_edges = []
#    for ee in range(len(edge_s)):
#        page_rank_pers_edges.append(page_rank_pers_nodes[edge_s[ee][0]][edge_s[ee][1]])
#
#    time_cost["PPR"] = (time.time() - start_time)
#    start_time = time.time()


    # local clustering coefficients for i and j (LCC_i, LCC_j)
    clust_nodes_obj = nx.clustering(G)
    clust_nodes = []
    for nn in range(len(nodes)):
        clust_nodes.append(clust_nodes_obj[nn])
    
    clust1_edges = []
    clust2_edges = []
    for ee in range(len(edge_s)):
        clust1_edges.append(clust_nodes[edge_s[ee][0]])
        clust2_edges.append(clust_nodes[edge_s[ee][1]])

    time_cost["LCC_i"] = (time.time() - start_time)
    start_time = time.time()
        
    # average neighbor degrees for i and j (AND_i, AND_j)
    ave_neigh_deg_nodes_obj = nx.average_neighbor_degree(G)
    ave_neigh_deg_nodes = []
    for nn in range(len(nodes)):
        ave_neigh_deg_nodes.append(ave_neigh_deg_nodes_obj[nn])
    
    ave_neigh_deg1_edges = []
    ave_neigh_deg2_edges = []
    for ee in range(len(edge_s)):
        ave_neigh_deg1_edges.append(ave_neigh_deg_nodes[edge_s[ee][0]])
        ave_neigh_deg2_edges.append(ave_neigh_deg_nodes[edge_s[ee][1]])
 
    time_cost["AND_i"] = (time.time() - start_time)
    start_time = time.time()
    
    # degree centralities for i and j (DC_i, DC_j)
    deg_cent_nodes_obj = nx.degree_centrality(G)
    deg_cent_nodes = []
    for nn in range(len(nodes)):
        deg_cent_nodes.append(deg_cent_nodes_obj[nn])
     
    deg_cent1_edges = []
    deg_cent2_edges = []
    for ee in range(len(edge_s)):
        deg_cent1_edges.append(deg_cent_nodes[edge_s[ee][0]])
        deg_cent2_edges.append(deg_cent_nodes[edge_s[ee][1]])

    time_cost["DC_i"] = (time.time() - start_time)
    start_time = time.time()

	# eigenvector centralities for i and j (EC_i, EC_j)
    tr = 1
    toler = 1e-6
    while tr == 1:
        try:
            eig_cent_nodes_obj = nx.eigenvector_centrality(G,tol = toler)
            tr = 0
        except:
            toler = toler*1e1
    
    eig_cent_nodes = []
    for nn in range(len(nodes)):
        eig_cent_nodes.append(eig_cent_nodes_obj[nn])
     
    eig_cent1_edges = []
    eig_cent2_edges = []
    for ee in range(len(edge_s)):
        eig_cent1_edges.append(eig_cent_nodes[edge_s[ee][0]])
        eig_cent2_edges.append(eig_cent_nodes[edge_s[ee][1]])

    time_cost["EC_i"] = (time.time() - start_time)
    start_time = time.time()

    # Katz centralities for i and j (KC_i, KC_j)
    ktz_cent_nodes_obj = nx.katz_centrality_numpy(G)
    ktz_cent_nodes = []
    for nn in range(len(nodes)):
        ktz_cent_nodes.append(ktz_cent_nodes_obj[nn])
    
    ktz_cent1_edges = []
    ktz_cent2_edges = []
    for ee in range(len(edge_s)):
        ktz_cent1_edges.append(ktz_cent_nodes[edge_s[ee][0]])
        ktz_cent2_edges.append(ktz_cent_nodes[edge_s[ee][1]])

    time_cost["KC_i"] = (time.time() - start_time)
    start_time = time.time()
        
    # Jaccard’s coefficient of neighbor sets of i, j (JC)
    jacc_coeff_obj = nx.jaccard_coefficient(G,edge_s)
    jacc_coeff_edges = []
    for uu,vv,jj in jacc_coeff_obj:
        jacc_coeff_edges.append([uu,vv,jj])   
    df_jacc_coeff = pd.DataFrame(jacc_coeff_edges, columns=['i','j','jacc_coeff'])
    df_jacc_coeff['ind'] = df_jacc_coeff.index

    time_cost["JC"] = (time.time() - start_time)
    start_time = time.time()


    # resource allocation index of i, j (RA)
    res_alloc_ind_obj = nx.resource_allocation_index(G, edge_s)
    res_alloc_ind_edges = []
    for uu,vv,jj in res_alloc_ind_obj:
        res_alloc_ind_edges.append([uu,vv,jj])
    df_res_alloc_ind = pd.DataFrame(res_alloc_ind_edges, columns=['i','j','res_alloc_ind'])    
    df_res_alloc_ind['ind'] = df_res_alloc_ind.index

    time_cost["RA"] = (time.time() - start_time)
    start_time = time.time()
  
  	# Adamic/Adar index of i, j (AA)
    adam_adar_obj =  nx.adamic_adar_index(G, edge_s)
    adam_adar_edges = []
    for uu,vv,jj in adam_adar_obj:
        adam_adar_edges.append([uu,vv,jj])
    df_adam_adar = pd.DataFrame(adam_adar_edges, columns=['i','j','adam_adar'])
    df_adam_adar['ind'] = df_adam_adar.index
    
    df_merge = pd.merge(df_jacc_coeff,df_res_alloc_ind, on=['ind','i','j'], sort=False)
    df_merge = pd.merge(df_merge,df_adam_adar, on=['ind','i','j'], sort=False)

    time_cost["AA"] = (time.time() - start_time)
    start_time = time.time()

    # preferential attachment (degree product) of i, j (PA)
    pref_attach_obj = nx.preferential_attachment(G, edge_s)
    pref_attach_edges = []
    for uu,vv,jj in pref_attach_obj:
        pref_attach_edges.append([uu,vv,jj])
    df_pref_attach = pd.DataFrame(pref_attach_edges, columns=['i','j','pref_attach'])
    df_pref_attach['ind'] = df_pref_attach.index

    time_cost["PA"] = (time.time() - start_time)
    start_time = time.time()
                
    # global features:
    # similarity of connections in the graph with respect to the node degree
    # degree assortativity (DA)
    deg_ass_net = nx.degree_assortativity_coefficient(G)

    time_cost["DA"] = (time.time() - start_time)
    start_time = time.time()

    # transitivity: fraction of all possible triangles present in G
    # network transitivity (clustering coefficient) (NT)
    transit_net = nx.transitivity(G)  
    # network diameter (ND)

    time_cost["NT"] = (time.time() - start_time)
    start_time = time.time()

    try:
        diam_net = nx.diameter(G)
    except:
        diam_net = np.inf
        
    ave_deg_net = [ave_deg_net for ii in range(len(edge_s))]
    var_deg_net = [var_deg_net for ii in range(len(edge_s))]
    ave_clust_net = [ave_clust_net for ii in range(len(edge_s))]
    deg_ass_net = [deg_ass_net for ii in range(len(edge_s))]
    transit_net = [transit_net for ii in range(len(edge_s))]
    diam_net = [diam_net for ii in range(len(edge_s))]
    com_ne = []
    for ee in range(len(edge_s)):
        com_ne.append(len(sorted(nx.common_neighbors(G,edge_s[ee][0],edge_s[ee][1]))))

    time_cost["global_stuff"] = (time.time() - start_time)
    start_time = time.time()
         
    # closeness centralities for i and j (CC_i, CC_j)
    closn_cent_nodes_obj = nx.closeness_centrality(G)
    closn_cent_nodes = []
    for nn in range(len(nodes)):
        closn_cent_nodes.append(closn_cent_nodes_obj[nn])
      
    closn_cent1_edges = []
    closn_cent2_edges = []
    for ee in range(len(edge_s)):
        closn_cent1_edges.append(closn_cent_nodes[edge_s[ee][0]])
        closn_cent2_edges.append(closn_cent_nodes[edge_s[ee][1]])

    time_cost["CC_i"] = (time.time() - start_time)
    start_time = time.time()
          
    # shortest path between i, j (SP)        
    short_Mat_aux = nx.shortest_path_length(G)
    short_Mat={}
    for ss in range(N):
        value = next(short_Mat_aux)
        short_Mat[value[0]] = value[1]   
    short_path_edges = []
    for ee in range(len(edge_s)):
        if edge_s[ee][1] in short_Mat[edge_s[ee][0]].keys():
            short_path_edges.append(short_Mat[edge_s[ee][0]][edge_s[ee][1]])  
        else:
            short_path_edges.append(np.inf)

    time_cost["SP"] = (time.time() - start_time)
    start_time = time.time()
            
    # load centralities for i and j (LC_i, LC_j)
    load_cent_nodes_obj = nx.load_centrality(G,normalized=True)
    load_cent_nodes = []
    for nn in range(len(nodes)):
        load_cent_nodes.append(load_cent_nodes_obj[nn])
    
    load_cent1_edges = []
    load_cent2_edges = []
    for ee in range(len(edge_s)):
        load_cent1_edges.append(load_cent_nodes[edge_s[ee][0]])
        load_cent2_edges.append(load_cent_nodes[edge_s[ee][1]])

    time_cost["LC_i"] = (time.time() - start_time)
    start_time = time.time()


    # shortest-path betweenness centralities for i and j (SPBC_i, SPBC_j)
    betw_cent_nodes_obj = nx.betweenness_centrality(G,normalized=True)
    betw_cent_nodes = []
    for nn in range(len(nodes)):
        betw_cent_nodes.append(betw_cent_nodes_obj[nn])
    
    betw_cent1_edges = []
    betw_cent2_edges = []
    for ee in range(len(edge_s)):
        betw_cent1_edges.append(betw_cent_nodes[edge_s[ee][0]])
        betw_cent2_edges.append(betw_cent_nodes[edge_s[ee][1]])
    
    
    time_cost["SPBC_i"] = (time.time() - start_time)
    start_time = time.time()   
        
    neigh_ = {}
    for nn in range(len(nodes)):
        neigh_[nn] = np.where(A[nn,:])[0]
    
    df_pref_attach = []
    for ee in range(len(edge_s)):
        df_pref_attach.append(len(neigh_[edge_s[ee][0]])*len(neigh_[edge_s[ee][1]]))
    
    U, sig, V = np.linalg.svd(A, full_matrices=False)
    S = np.diag(sig)
    Atilda = np.dot(U, np.dot(S, V))
    Atilda = np.array(Atilda)
    
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # entry i, j in low rank approximation (LRA) via singular value decomposition (SVD)
    svd_edges = []
    # dot product of columns i and j in LRA via SVD for each pair of nodes i, j
    svd_edges_dot = []
    # average of entries i and j’s neighbors in low rank approximation
    svd_edges_mean = []
    for ee in range(len(edge_s)):
        svd_edges.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]]))        
 
    time_cost["SVD"] = (time.time() - start_time)
    start_time = time.time()
    
    # Leicht-Holme-Newman index of neighbor sets of i, j (LHN)
    f_LHN = lambda num,den: 0 if (num==0 and den==0) else float(num)/den 
    LHN_edges = [f_LHN(num,den) for num,den in zip(np.array(com_ne),np.array(df_pref_attach))]
    
    U, sig, V = np.linalg.svd(A)
    S = linalg.diagsvd(sig, A.shape[0], A.shape[1])
    S_trunc = S.copy()
    S_trunc[S_trunc < sig[int(np.ceil(np.sqrt(A.shape[0])))]] = 0
    Atilda = np.dot(np.dot(U, S_trunc), V)
    Atilda = np.array(Atilda)

    time_cost["LHN"] = (time.time() - start_time)
    start_time = time.time()
   
    f_mean = lambda x: np.mean(x) if len(x)>0 else 0
    # an approximation of LRA (LRA-approx)
    svd_edges_approx = []
    # an approximation of dLRA (dLRA-approx)
    svd_edges_dot_approx = []
    # an approximation of mLRA (mLRA-approx)
    svd_edges_mean_approx = []
    for ee in range(len(edge_s)):
        svd_edges_approx.append(Atilda[edge_s[ee][0],edge_s[ee][1]])
        svd_edges_dot_approx.append(np.inner(Atilda[edge_s[ee][0],:],Atilda[:,edge_s[ee][1]]))
        svd_edges_mean_approx.append(f_mean(Atilda[edge_s[ee][0],neigh_[edge_s[ee][1]]])) 
    
    time_cost["dLRA"] = (time.time() - start_time)
    start_time = time.time()

    # number of nodes (N)
    num_nodes = A.shape[0]
    # number of observed edges (OE)
    num_edges = int(np.sum(A)/2)
    
    # construct a dictionary of the features
    d = {'i':edge_pairs_f_i, 'j':edge_pairs_f_j, 'com_ne':com_ne, 'ave_deg_net':ave_deg_net, \
         'var_deg_net':var_deg_net, 'ave_clust_net':ave_clust_net, 'num_triangles_1':numtriang1_edges, 'num_triangles_2':numtriang2_edges, \
         'pag_rank1':page_rank1_edges, 'pag_rank2':page_rank2_edges, 'clust_coeff1':clust1_edges, 'clust_coeff2':clust2_edges, 'ave_neigh_deg1':ave_neigh_deg1_edges, 'ave_neigh_deg2':ave_neigh_deg2_edges,\
         'eig_cent1':eig_cent1_edges, 'eig_cent2':eig_cent2_edges, 'deg_cent1':deg_cent1_edges, 'deg_cent2':deg_cent2_edges, 'clos_cent1':closn_cent1_edges, 'clos_cent2':closn_cent2_edges, 'betw_cent1':betw_cent1_edges, 'betw_cent2':betw_cent2_edges, \
         'load_cent1':load_cent1_edges, 'load_cent2':load_cent2_edges, 'ktz_cent1':ktz_cent1_edges, 'ktz_cent2':ktz_cent2_edges, 'pref_attach':df_pref_attach, 'LHN':LHN_edges, 'svd_edges':svd_edges,'svd_edges_dot':svd_edges_dot,'svd_edges_mean':svd_edges_mean,\
         'svd_edges_approx':svd_edges_approx,'svd_edges_dot_approx':svd_edges_dot_approx,'svd_edges_mean_approx':svd_edges_mean_approx, 'short_path':short_path_edges, 'deg_assort':deg_ass_net, 'transit_net':transit_net, 'diam_net':diam_net, \
         'num_nodes':num_nodes, 'num_edges':num_edges}     
    
    # construct a dataframe of the features
    df_feat = pd.DataFrame(data=d)
    df_feat['ind'] = df_feat.index
    df_feat = pd.merge(df_feat,df_merge, on=['ind','i','j'], sort=False)
    return df_feat, time_cost



####################################
# 12/17 revised revised prediction method, just predict what is going on assuming
# everything is lost in the last layer (i.e. have no information for the target layer)
# ###################################


def mxx(edges_orig):
    n_layers = len(edges_orig)
    temp_max = np.max(edges_orig[0])
    for i in range(n_layers):
        temp_max = max(temp_max,np.max(edges_orig[i]))
    return temp_max
    
    
def topol_stacking_temporal_4(edges_orig, target_layer, predict_num,name): 
    
    """ 
    Assuming we have no information at all for the target layer
    """
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
    
    
    #### load target layer A
    
    mymax= mxx(edges_orig)
    num_nodes = int(max( np.max(target_layer), mymax)) +1
    #num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    

    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
    #A_tr.append(A_train)
    #### extract features #####
    #sample_true_false_edges_revised_3(A, A_true, A_tr, A_hold, predict_num, name)

    
    
    sample_true_false_edges_revised_4(A, A_tr, predict_num, name)
    #sample_tf_6_allmiss(A, A_tr, predict_num)
    #sample_tf_3_allmiss(A, A_tr, predict_num)
    #a,b,c,d = sample_tf_4_allmiss(A, A_tr, predict_num)
    
    #return a,b,c,d


    
    for i in range(predict_num, len(A_tr)):
        edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt").astype('int'))
        edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt").astype('int'))
        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr[i-predict_num :i], edge_f_tr[i-predict_num])
        df_f_tr.append(df_temp)
        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr[i-predict_num :i], edge_t_tr[i-predict_num])
        df_t_tr.append(df_temp)
    
    
    edge_t_ho= np.loadtxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt").astype('int')
    edge_f_ho= np.loadtxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt").astype('int')
    df_f_ho, time1 = gen_topol_feats_3(A, A_tr[ -(predict_num):], edge_f_ho)
    df_t_ho, time2 = gen_topol_feats_3(A, A_tr[ -(predict_num):], edge_t_ho)
    

#    return df_f_tr, df_t_tr, df_f_ho
    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns    
    
    v1 = df_t_tr[0].values
    v2 = df_f_tr[0].values

    
    # Stack Together for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    # Let the randomforest make selection
    
    for i in range(1, len(A_tr)-predict_num):
        v1 = np.vstack((v1,df_t_tr[i].values))
        v2 = np.vstack((v2,df_f_tr[i].values))
    
#    return df_t_tr, df_f_tr, df_t_ho, df_f_ho  
    df_t_tr_ = pd.DataFrame(data=v1, columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2, columns = df_f_tr_columns)
    
    
    
    
    
    column_name = list(df_t_tr_.columns)
    
    for j in range(1,predict_num+1):  
        for i in range (44*j, 44*(j+1)):
            column_name[i] = column_name[i]+"_"+str(j)
    
    print(column_name)
    
    df_t_tr_.columns = column_name
    df_f_tr_.columns = column_name
    df_t_ho.columns = column_name
    df_f_ho.columns = column_name
    
    feat_path = "./ef_gen_tr/"+str(name)+"/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t')
    df_f_tr_.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"+str(name)+"/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #return df_t_tr_,df_f_tr_,df_t_ho,df_f_ho
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set_3(df_t_tr_,df_f_tr_,predict_num)
    df_ho = creat_full_set_3(df_t_ho,df_f_ho,predict_num)
    
    feats = list(df_tr.columns)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'+"/"+str(name)  # output path
    creat_numpy_files_3(dir_output, df_ho, df_tr,predict_num)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' +"/"+str(name)
    path_to_results = './results'+"/"+str(name)
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    #auprc, auc,precision,recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    auprc, auc,precision,recall, featim = heldout_performance_bestchoice(path_to_data, path_to_results, n_depth, n_est)
    print("NAME IS ", name)

    return auprc, auc, precision, recall, featim, feats
#, mytime
    
def sample_true_false_edges_revised_4(A_orig, A_train, predict_num, name):  
    
    """ 
    For each edge pair,it sample not only from the last layer that we wanted to predict
    but also from previous layers that might have useful information in them i.e. basically
    if we are given 9 layers before as information, and we consider the features of 4 layers
    than we are saying that we want to sample from the 7 choices and only consider the last
    of those 4 layers.
    1,2,3,4,5,6,7,8,9,10
    10 being the layer that needs to be predicted. Take 4 layer of information
    for example: 4,5,6,7
    than we consider the label of the edge true edge if it appears in 7 and false if not
    After that we consider take the features from all 4,5,6,7 and concatenate them
    
    Previously if we sample 10000 edges than we have 10000x45 as our output feature vector
    Now we have 10000 x (45*4) i,e, there are 180 features for 1 pair of edge.
    
    
    Note that for the target layer, we have no information at all for the target layer
    
    
    """
    
    train_num = len(A_train)
    
    Nsamples = 10000
    nsim_id = 0
    np.random.seed(nsim_id)
    num_sample_edges = math.ceil(Nsamples/(len(A_train)-predict_num))-1
    
    
    A_edge_t = []
    A_edge_f = []
    for i in range(predict_num, len(A_train)):
        nodes, edges = adj_to_nodes_edges(A_train[i])
        
        pos_edges = sparse.find(sparse.triu(A_train[i],1)) # true candidates
        A_neg = -1*A_train[i] + 1
        neg_edges = sparse.find(sparse.triu(A_neg,1)) # false candidates

        temp_edge_t = [] # list of true edges (positive samples)
        temp_edge_f = [] # list of false edges (negative samples)
        for ll in range(num_sample_edges):
    	    edge_t_idx_aux = np.random.randint(len(pos_edges[0]))
    	    edge_f_idx_aux = np.random.randint(len(neg_edges[0]))
    	    temp_edge_t.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux]))
    	    temp_edge_f.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux]))
        
        A_edge_t.append(temp_edge_t)
        A_edge_f.append(temp_edge_f)
        # store for later use
        if not os.path.isdir("./edge_tf_tr/"):
    	    os.mkdir("./edge_tf_tr/")
        np.savetxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt",temp_edge_t,fmt='%u')
        np.savetxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt",temp_edge_f,fmt='%u')
        
    

    A_diff = A_orig
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A_orig + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_tt = [] # list of true edges (positive samples)
    edge_ff = [] # list of false edges (negative samples)
    for ll in range(Nsamples):
        edge_tt_idx_aux = np.random.randint(len(e_diff[0]))
        edge_ff_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_tt.append((e_diff[0][edge_tt_idx_aux],e_diff[1][edge_tt_idx_aux]))
        edge_ff.append((ne_orig[0][edge_ff_idx_aux],ne_orig[1][edge_ff_idx_aux]))
    
    # store for later use
    if not os.path.isdir("./edge_tf_true/"):
        os.mkdir("./edge_tf_true/")
    np.savetxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt",edge_tt,fmt='%u')
    np.savetxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt",edge_ff,fmt='%u')
    
##################
"""
Add different sample methods to the new sampling form
"""
"""
WE WANT TO USE DIFFERNET NUMBER OF TOPOLOGICAL FEATURES TO GENERATE SOME PRETTY PLOTS
"""
def topol_stacking_temporal_4_different(edges_orig, target_layer, predict_num,name, feats): 
    """ 
    Assuming we have no information at all for the target layer
    """
    
    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []
    
    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []
    
    
    #### load target layer A
    
    mymax= mxx(edges_orig)
    num_nodes = int(max( np.max(target_layer), mymax)) +1
    #num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))
    
    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()
    

    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))
        
        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        
        
    #A_tr.append(A_train)
    #### extract features #####
    #sample_true_false_edges_revised_3(A, A_true, A_tr, A_hold, predict_num, name)

    
    
    sample_true_false_edges_revised_4(A, A_tr, predict_num,name)
    #sample_tf_6_allmiss(A, A_tr, predict_num)
    #sample_tf_3_allmiss(A, A_tr, predict_num)
    #a,b,c,d = sample_tf_4_allmiss(A, A_tr, predict_num)
    
    #return a,b,c,d


    
    for i in range(predict_num, len(A_tr)):
        edge_t_tr.append(np.loadtxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt").astype('int'))
        edge_f_tr.append(np.loadtxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt").astype('int'))
        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr[i-predict_num :i], edge_f_tr[i-predict_num])
        df_f_tr.append(df_temp)
        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr[i-predict_num :i], edge_t_tr[i-predict_num])
        df_t_tr.append(df_temp)
    
    
    edge_t_ho= np.loadtxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt").astype('int')
    edge_f_ho= np.loadtxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt").astype('int')
    df_f_ho, time1 = gen_topol_feats_3(A, A_tr[ -(predict_num):], edge_f_ho)
    df_t_ho, time2 = gen_topol_feats_3(A, A_tr[ -(predict_num):], edge_t_ho)
    

#    return df_f_tr, df_t_tr, df_f_ho
    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns    
    
    v1 = df_t_tr[0].values
    v2 = df_f_tr[0].values

    
    # Stack Together for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    # Let the randomforest make selection
    
    for i in range(1, len(A_tr)-predict_num):
        v1 = np.vstack((v1,df_t_tr[i].values))
        v2 = np.vstack((v2,df_f_tr[i].values))
    
#    return df_t_tr, df_f_tr, df_t_ho, df_f_ho  
    df_t_tr_ = pd.DataFrame(data=v1, columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2, columns = df_f_tr_columns)
    
    
    
    
    
    column_name = list(df_t_tr_.columns)
    
    for j in range(1,predict_num+1):  
        for i in range (44*j, 44*(j+1)):
            column_name[i] = column_name[i]+"_"+str(j)
    
    
    df_t_tr_.columns = column_name
    df_f_tr_.columns = column_name
    df_t_ho.columns = column_name
    df_f_ho.columns = column_name
    
    feat_path = "./ef_gen_tr/"+str(name)+"/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t')
    df_f_tr_.to_pickle(feat_path + 'df_f')
    
    feat_path = "./ef_gen_ho/"+str(name)+"/"
    if not os.path.isdir(feat_path):
        os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t')
    df_f_ho.to_pickle(feat_path + 'df_f')
    
    
    #return df_t_tr_,df_f_tr_,df_t_ho,df_f_ho
    
    #### load dataframes for train and holdout sets ####
    df_tr = creat_full_set_3(df_t_tr_,df_f_tr_,predict_num)
    df_ho = creat_full_set_3(df_t_ho,df_f_ho,predict_num)
    
    
    
    feature_set = ['com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net',
           'num_triangles_1', 'num_triangles_2', 
           'pag_rank1', 'pag_rank2', 'clust_coeff1', 'clust_coeff2',
           'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2',
           'deg_cent1', 'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1',
           'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2',
           'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean',
           'svd_edges_approx', 'svd_edges_dot_approx', 'svd_edges_mean_approx',
           'short_path', 'deg_assort', 'transit_net', 'diam_net',
           'jacc_coeff', 'res_alloc_ind', 'adam_adar' , 'num_nodes','num_edges']  
    
    
    #return df_t_tr_,df_f_tr_,df_t_ho,df_f_ho
    
    # choose a subset of features want to test on 
    feature_set = np.array(feature_set)
    sub = random.sample(range(41), feats)
    feature_subset = feature_set[sub]
    
    temp_col = []
    
    for feat in feature_subset:
        c = [s for s in column_name if s.startswith(feat)]
        temp_col.append(c)
     
    new_columns = [item for sublist in temp_col for item in sublist]
    
    new = [x for x in new_columns]
    
    new.append("i")
    new.append("j")
    new.append("TP")
    
    df_tr_sub = df_tr[new]
    df_ho_sub = df_ho[new]
    

    #### creat and save feature matrices #### 
    dir_output = './feature_metrices' +"/"+str(name) # output path
    creat_numpy_files_4(dir_output, df_ho_sub, df_tr_sub,predict_num,feature_subset)
    
    
    #### perform model selection #### 
    path_to_data = './feature_metrices' +"/"+str(name)
    path_to_results = './results'+"/"+str(name)
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    n_depth, n_est = model_selection(path_to_data, path_to_results, n_depths, n_ests)
    
    
    #### perform model selection #### 
    auprc, auc,precision,recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    print("NAME IS ", name)

    return auprc, auc, precision, recall
#, mytime
    
def creat_numpy_files_4(dir_results, df_ho, df_tr, predict_num, feature_subset):
    
    """ 
    This functiion multiplied the column number and attache verything that is important in the function
    """
    feature_set = feature_subset

    #'page_rank_pers_edges', removed
    
    
    full_feat_set = [None] * (predict_num*len(feature_set))
    len_of_feat = len(feature_set)
    
    for k in range(len_of_feat):
        full_feat_set[k] = feature_set[k]
    
    for j in range(1,predict_num):  
        for i in range (len_of_feat*j, len_of_feat*(j+1)):
            full_feat_set[i] = feature_set[i-len_of_feat*j]+"_"+str(j)
        
    
    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    
    
    X_train_orig = df_tr
    y_train_orig = np.array(df_tr.TP)

    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        

        X_train = train.loc[:,full_feat_set]
        X_test = test.loc[:,full_feat_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=len_of_feat)
        
        #X_train = X_train.loc[:,~X_train.columns.duplicated()]
        #y_train = y_train.loc[:,~y_train.columns.duplicated()]
        
        X_train, y_train = sm.fit_resample(X_train.as_matrix(), y_train.as_matrix())

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,full_feat_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  
    
    #X_seen = X_seen.loc[:,~X_seen.columns.duplicated()]
    
    # balance train set with upsampling
    sm = RandomOverSampler(random_state=len_of_feat)
    X_seen, y_seen = sm.fit_resample(X_seen.as_matrix(), y_seen.as_matrix())
    
    

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,full_feat_set]
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")
    
def heldout_performance_bestchoice(path_to_data, path_to_results, n_depth, n_est):
    
    """ 
    This function trains a random forest model on seen data and performs prediction on heldout.

    Input and Parameters:
    -------
    path_to_data: path to held out featute matrices 
    path_to_results: path to save model performance ast txt file
    n_depth: max_depth for random forest parameter
    n_est: n_estimators for random forest parameter

    Returns:
    -------
    auc_measure: auc on heldout
    precision_total: precision of positive class on heldout
    recall_total: recall of positive class on heldout

    Examples:
    -------
    auc , precision, recall = heldout_performance(path_to_data, path_to_results, n_depth, n_est)
    """
    
    if not os.path.isdir(path_to_results):
        os.mkdir(path_to_results)
    f = open(path_to_results + '/RF_Best_metrics.txt','w')
    #path_to_data = './feature_metrices'
    
    # read data
    X_train = np.load(path_to_data+'/X_Eseen.npy')
    y_train = np.load(path_to_data+'/y_Eseen.npy')
    X_test = np.load(path_to_data+'/X_Eunseen.npy')
    y_test = np.load(path_to_data+'/y_Eunseen.npy')
    
    
    col_mean = np.nanmean(X_train, axis=0)
    inds = np.where(np.isnan(X_train))
    X_train[inds] = np.take(col_mean, inds[1])
    
    col_mean = np.nanmean(X_test, axis=0)
    inds = np.where(np.isnan(X_test))
    X_test[inds] = np.take(col_mean, inds[1])
     
       
    # train the model
    dtree_model = RandomForestClassifier(n_estimators=n_est,max_depth=n_depth).fit(X_train, y_train)
    
    
    
    # feature importance and prediction on test set 
    feature_importance = dtree_model.feature_importances_
    dtree_predictions = dtree_model.predict(X_test)
    dtree_proba = dtree_model.predict_proba(X_test)
      
    # calculate performance metrics
    cm_dt4 = confusion_matrix(y_test, dtree_predictions)
    auc_measure = roc_auc_score(y_test, dtree_proba[:,1])
    auprc = average_precision_score(y_test, dtree_proba[:,1])
    precision = precision_score(y_test, dtree_predictions, average = "micro")
    recall = recall_score(y_test, dtree_predictions, average = "micro")
    precision_1 = precision_score(y_test, dtree_predictions, average = "macro")
    recall_1 = recall_score(y_test, dtree_predictions, average = "macro")
    
    precision_total, recall_total, f_measure_total, _ = precision_recall_fscore_support(y_test, dtree_predictions, average=None)

    
    #pre = sklearn.metrics.precision_score(y_test, dtree_predictions)
    #rec = sklearn.metrics.recall_score(y_test, dtree_predictions)
    
    
    
    
    f.write('heldout_AUC = '+ str(auc_measure)+'\n')
    f.write('heldout_precision = '+ str(precision_total)+'\n')
    f.write('heldout_recall = '+ str(recall_total)+'\n')
    f.write('heldout_f_measure = '+ str(f_measure_total)+'\n')
    f.write('feature_importance = '+ str(list(feature_importance))+'\n')
    f.close()
    
    precision_return = []
    recall_return = []
    
    print("AUC: " +str(np.round(auc_measure,2)))
    print("precision: " +str(precision))
    print("recall: " +str(recall))
    
    precision_return.append(precision)
    recall_return.append(recall)
    precision_return.append(precision_1)
    recall_return.append(recall_1)
    precision_return.append(np.mean(precision_total))
    recall_return.append(np.mean(recall_total))

    print("precision again: " + str(precision_1))
    print("recall again: " + str(recall_1))
    print("precision again: " + str(np.mean(precision_total)))
    print("recall again: " + str(np.mean(recall_total)))

    return auprc, auc_measure, precision_return, recall_return, feature_importance



""
def sbm_model(train, test, p,T):
    ################
    #Default value of K,L,S,R, sampling, printp, for general function purposes.
    ##########
    
    K = 4
    L = 4
    S = 2
    R = 2 # either it is a true edge or it is a false edge
    sampling = 2
    
    fh=open(train,'r')
    igot=fh.readlines()
    trainn=[]
    
    for line in igot:
        about = line.strip().split(' ')
        trainn.append((int(about[0]),int(about[1]),int(about[2]),int(about[3])))
    fh.close()
    
    fh2=open(test,'r')
    igot2=fh2.readlines()
    testt={}
    praij={}
    for line in igot2:
        about = line.strip().split(' ')
        testt[(int(about[0]),int(about[1]),int(about[2]))]=int(about[3])
        praij[(int(about[0]),int(about[1]),int(about[2]))]=[0.]*R
    fh2.close()
    ""
    for w in range(sampling):
        theta=[]
        ntheta=[]
        for i in range(p):
            vec=[]
            for k in range(K):
                vec.append(random.random())
            theta.append(vec)
            ntheta.append([0.0]*K)
        tau=[]
        ntau=[]
        for t in range(T):
            vec=[]
            for s in range(S):
                vec.append(random.random())
            tau.append(vec)
            ntau.append([0.0]*S)
        pr=[]
        npr=[]
        for k in range(K):
            pr.append([])
            npr.append([])
            for l in range(L):
                pr[k].append([])
                npr[k].append([])
        for k in range(K):
            for z in range(L-(k)):
                l=k+z
                if k == l:
                    vec=[]
                    b=[]
                    for s in range(S):
                        a=[]
                        for r in range(R):
                            a.append(random.random())
                        vec.append(a)
                        b.append([0.]*R)
                    pr[k][l] = vec
                    npr[k][l]=b
                else:
                    vec=[]
                    b=[]
                    for s in range(S):
                        a=[]
                        for r in range(R):
                            a.append(random.random())
                        vec.append(a)
                        b.append([0.]*R)
                    pr[k][l]=vec
                    pr[l][k]=vec
                    npr[k][l]=b
                    npr[l][k]=b

        #Normalizations:
        for i in range(p):
            D=0.
            for k in range(K):
                D=D+theta[i][k]
            for k in range(K):
                theta[i][k]=theta[i][k]/D

        for t in range(T):
            D=0.
            for s in range(S):
                D=D+tau[t][s]
            for s in range(S):
                tau[t][s]=tau[t][s]/D

        for k in range(K):
            for l in range(L):
                for a in range(S):
                    D=0.
                    for r in range(R):
                        D=D+pr[k][l][s][r]
                    for r in range(R):
                        pr[k][l][s][r]=pr[k][l][s][r]/D

        #########################################################################################
        Runs=1000
        #Al=1.
        #While Al>0.00000001:
        for g in range(Runs):
            for e in trainn:
                
                ############## There is BUG in original code
                t=int(e[2])
                n1=int(e[0])
                n2=int(e[1])
                ra=int(e[3])
                D=0.	
                for s in range(S):
                    for l in range(L):
                        for k in range(K):
                            D=D+theta[n1][k]*theta[n2][l]*tau[t][s]*pr[k][l][s][ra]
                for s in range(S):
                    for l in range(L):
                        for k in range(K):
                            a=(theta[n1][k]*theta[n2][l]*tau[t][s]*pr[k][l][s][ra])/D
                            ntheta[n1][k]=ntheta[n1][k]+a
                            ntheta[n2][l]=ntheta[n2][l]+a
                            ntau[t][s]=ntau[t][s]+a
                            npr[k][l][s][ra]=npr[k][l][s][ra]+a

            #Normalizations:
            err=0.
            for i in range(p):
                D=0.
                for k in range(K):
                    D=D+ntheta[i][k]
                for k in range(K):
                    ntheta[i][k]=ntheta[i][k]/(D+0.00000000001)

            for t in range(T):
                D=0.
                for s in range(S):
                    D=D+ntau[t][s]
                for s in range(S):
                    ntau[t][s]=ntau[t][s]/(D+0.0000000000001)

            for k in range(K):
                for l in range(L):
                    for s in range(S):
                        D=0.
                        for r in range(R):
                            D=D+npr[k][l][s][r]
                        for r in range(R):
                            npr[k][l][s][r]=npr[k][l][s][r]/(D+0.000000001)

            theta=copy.copy(ntheta)
            tau=copy.copy(ntau)
            for k in range(K):
                for l in range(L):
                    for s in range(S):
                        pr[k][l][s]=npr[k][l][s]
            for i in range(p):
                ntheta[i]=[0.]*K
            for t in range(T):
                ntau[t]=[0.]*S
            for k in range(K):
                for l in range(L):
                    for s in range(S):
                        npr[k][l][s]=[0.]*R

        Like=0.
        for e in trainn:
            t=int(e[2])
            n1=int(e[0])
            n2=int(e[1])
            ra=int(e[3])
            D=0.
            for s in range(S):
                for l in range(L):
                    for k in range(K):
                        D=D+theta[n1][k]*theta[n2][l]*tau[t][s]*pr[k][l][s][ra]
            Like=Like+np.log(D)

        #scores:
        for e in testt:
            t=int(e[2])
            n1=int(e[0])
            n2=int(e[1])
            pra=0.
            for rr in range(R):
                pra=0.
                for s in range(S):
                    for k in range(K):
                        for l in range(L):
                            pra=pra+theta[n1][k]*theta[n2][l]*tau[t][s]*pr[k][l][s][rr]
                praij[(n1,n2,t)][rr]=praij[(n1,n2,t)][rr]+pra/sampling
            
    ####################
    # # Test scores
    # ##################
    
    results = []
    for line in igot2:
        about = line.strip().split(' ')
        rrrrr = praij[(int(about[0]),int(about[1]),int(about[2]))] 
        results.append(rrrrr)
    fh2.close()
    

    return results
    


""
def sample_true_false_edges_revised_5(A, A_ho, A_tr, A_tr_new, predict_num,name):  
    
    """ 
    Transfer output to be an input for the SBM model
    """
    
    Nsamples = 10000
    nsim_id = 0
    np.random.seed(nsim_id)
    num_sample_edges = math.ceil(Nsamples/(len(A_tr)-predict_num))-1
    
    
    A_edge_t = []
    A_edge_f = []
    
    #if not os.path.isdir("./for_sbm/"):
    try:
        os.mkdir("./for_sbm/")
    except:
        print("")
    for i in range(predict_num):
        A_diff = A_tr[i]
        nodes, edges = adj_to_nodes_edges(A_tr_new[i])
        
        pos_edges = sparse.find(sparse.triu(A_tr[i],1)) # true candidates
        A_neg = -1*A_tr[i] + 1
        neg_edges = sparse.find(sparse.triu(A_neg,1)) # false candidates
        test_lay = []
        
        for ll in range(num_sample_edges):
            edge_t_idx_aux = np.random.randint(len(pos_edges[0]))
            edge_f_idx_aux = np.random.randint(len(neg_edges[0]))
            test_lay.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux], predict_num, 1))
            test_lay.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux], predict_num, 0))
    
        np.savetxt("./for_sbm/temp_train_{}".format(i)+"_"+str(name)+".txt", test_lay, fmt='%u') 
    
    for i in range(predict_num, len(A_tr)):
        A_diff = A_tr[i] - A_ho[i-predict_num]
        nodes, edges = adj_to_nodes_edges(A_diff)
        pos_edges = sparse.find(sparse.triu(A_tr[i],1)) # true candidates
        A_neg = -1*A_tr[i] + 1
        neg_edges = sparse.find(sparse.triu(A_neg,1)) # false candidates

        temp_edge_t = [] # list of true edges (positive samples)
        temp_edge_f = [] # list of false edges (negative samples)
        test_lay = []
        for ll in range(num_sample_edges):
            edge_t_idx_aux = np.random.randint(len(pos_edges[0]))
            edge_f_idx_aux = np.random.randint(len(neg_edges[0]))
            temp_edge_t.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux]))
            temp_edge_f.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux]))
            test_lay.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux], predict_num, 1))
            test_lay.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux], predict_num, 0))
        try:
            os.mkdir("./edge_tf_tr/")
        except:
            print("")
        #if not os.path.isdir("./edge_tf_tr/"):
            #os.mkdir("./edge_tf_tr/")
        np.savetxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt",temp_edge_t,fmt='%u')
        np.savetxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt",temp_edge_f,fmt='%u')
        np.savetxt("./for_sbm/temp_train_{}".format(i)+"_"+str(name)+".txt", test_lay, fmt='%u')
  
    
    
    for i in range(len(A_tr_new)-1):
        A_diff = A_ho[i] - A_tr_new[i]
        nodes, edges = adj_to_nodes_edges(A_tr_new[i])
        
        pos_edges = sparse.find(sparse.triu(A_tr_new[i],1)) # true candidates
        A_neg = -1*A_ho[i] + 10
        neg_edges = sparse.find(sparse.triu(A_neg,1)) # false candidates
        test_lay = []
        
        for ll in range(num_sample_edges):
            edge_t_idx_aux = np.random.randint(len(pos_edges[0]))
            edge_f_idx_aux = np.random.randint(len(neg_edges[0]))
            test_lay.append((pos_edges[0][edge_t_idx_aux],pos_edges[1][edge_t_idx_aux], predict_num, 1))
            test_lay.append((neg_edges[0][edge_f_idx_aux],neg_edges[1][edge_f_idx_aux], predict_num, 0))    
        np.savetxt("./for_sbm/test_{}".format(i+predict_num)+"_"+str(name)+".txt", test_lay, fmt='%u')
        

    A_diff = A_ho[-1] - A_tr_new[-1]
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_ho_aux = -1*A_ho[-1] + 1
    ne_ho = sparse.find(sparse.triu(A_ho_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_t = [] # list of true edges (positive samples)
    edge_f = [] # list of false edges (negative samples)
    train_final = []
    
    for ll in range(Nsamples):
        edge_t_idx_aux = np.random.randint(len(e_diff[0]))
        edge_f_idx_aux = np.random.randint(len(ne_ho[0]))
        edge_t.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux]))
        edge_f.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux]))
        train_final.append((e_diff[0][edge_t_idx_aux],e_diff[1][edge_t_idx_aux], predict_num, 1))
        train_final.append((ne_ho[0][edge_f_idx_aux],ne_ho[1][edge_f_idx_aux], predict_num, 0))
    # store for later use
    try:
        os.mkdir("./edge_tf_tr/")
    except:
        print("")
    #if not os.path.isdir("./edge_tf_tr/"):
        #os.mkdir("./edge_tf_tr/")
    np.savetxt("./edge_tf_tr/edge_t_final"+"_"+str(name)+".txt",edge_t,fmt='%u')
    np.savetxt("./edge_tf_tr/edge_f_final"+"_"+str(name)+".txt",edge_f,fmt='%u')
    np.savetxt("./for_sbm/train_final"+"_"+str(name)+".txt",train_final,fmt='%u')
    
    
    
    A_diff = A - A_ho[-1]
    e_diff = sparse.find(sparse.triu(A_diff,1)) # true candidates
    A_orig_aux = -1*A + 1
    ne_orig = sparse.find(sparse.triu(A_orig_aux,1)) # false candidates
    Nsamples = 10000 # number of samples
    edge_tt = [] # list of true edges (positive samples)
    edge_ff = [] # list of false edges (negative samples)
    test_final = []
    
    for ll in range(Nsamples):
        edge_tt_idx_aux = np.random.randint(len(e_diff[0]))
        edge_ff_idx_aux = np.random.randint(len(ne_orig[0]))
        edge_tt.append((e_diff[0][edge_tt_idx_aux],e_diff[1][edge_tt_idx_aux]))
        edge_ff.append((ne_orig[0][edge_ff_idx_aux],ne_orig[1][edge_ff_idx_aux]))
        test_final.append((e_diff[0][edge_tt_idx_aux],e_diff[1][edge_tt_idx_aux], predict_num, 1))
        test_final.append((ne_orig[0][edge_ff_idx_aux],ne_orig[1][edge_ff_idx_aux], predict_num, 0))
    # store for later use
    try:
        os.mkdir("./edge_tf_true/")
    except:
        print("")
    #if not os.path.isdir("./edge_tf_true/"):
        #os.mkdir("./edge_tf_true/")
    np.savetxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt",edge_tt,fmt='%u')
    np.savetxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt",edge_ff,fmt='%u')        
    np.savetxt("./for_sbm/test_final"+"_"+str(name)+".txt",test_final,fmt='%u')
        
        
##################

""


""
def creat_full_set_final(df_t,df_f,predict_num):
    
    df_t = df_t.drop_duplicates(subset=['i','j'], keep="first")
    df_f = df_f.drop_duplicates(subset=['i','j'], keep="first")

    df_t.insert(2, "TP", 1, True)
    df_f.insert(2, "TP", 0, True)
    
    df_all = [df_t, df_f]
    df_all = pd.concat(df_all)
    
    df_all.loc[df_all['short_path'] == np.inf,'short_path'] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'],default=0)
    df_all.loc[df_all['diam_net'] == np.inf,'diam_net'] = 1e6
    
    # data cleaning
    for i in range(1,predict_num+1):
        df_all.loc[df_all['short_path_{}'.format(i)] == np.inf,'short_path_{}'.format(i)] = 1000*max(df_all.loc[~(df_all['short_path'] == np.inf),'short_path'],default=0)
        df_all.loc[df_all['diam_net_{}'.format(i)] == np.inf,'diam_net_{}'.format(i)] = 1e6
     
    return df_all



def creat_numpy_files_final(dir_results, df_ho, df_tr, predict, sbm, lstm ,ts):
    
    """ 
    This functiion multiplied the column number and attache verything that is important in the function
    """
    
    feature_set = ['com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net',
           'num_triangles_1', 'num_triangles_2', 
           'pag_rank1', 'pag_rank2', 'clust_coeff1', 'clust_coeff2',
           'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2',
           'deg_cent1', 'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1',
           'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2',
           'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean',
           'svd_edges_approx', 'svd_edges_dot_approx', 'svd_edges_mean_approx',
           'short_path', 'deg_assort', 'transit_net', 'diam_net',
           'jacc_coeff', 'res_alloc_ind', 'adam_adar' , 'num_nodes','num_edges']  
    #'page_rank_pers_edges', removed
    
    
    full_feat_set = [None] * ((predict+1)*len(feature_set))
    len_of_feat = len(feature_set)
    
    for k in range(len_of_feat):
        full_feat_set[k] = feature_set[k]
    
    for j in range(1,predict+1):  
        for i in range (len_of_feat*j, len_of_feat*(j+1)):
            full_feat_set[i] = feature_set[i-len_of_feat*j]+"_"+str(j)
        
    #print(full_feat_set)
    
    if sbm==1:
        full_feat_set.append("sbm")
    
    if lstm==1:
        full_feat_set.append("lstm")
    
    
    X_test_heldout = df_ho
    y_test_heldout = np.array(df_ho.TP)
    
    
    X_train_orig = df_tr
    y_train_orig = np.array(df_tr.TP)

    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    try:
        os.mkdir(dir_results+'/')
    except:
        print("")
    #if not os.path.isdir(dir_results+'/'):
    #    os.mkdir(dir_results+'/')
    
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
         
        train = X_train_orig.iloc[np.array(cv_train)]
        test = X_train_orig.iloc[np.array(cv_test)]

        y_train = train.TP
        y_test = test.TP
        

        X_train = train.loc[:,full_feat_set]
        X_test = test.loc[:,full_feat_set]

        X_test.fillna(X_test.mean(), inplace=True)
        X_train.fillna(X_train.mean(), inplace=True)

        sm = RandomOverSampler(random_state=len_of_feat)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), X_train)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), y_train)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), X_test)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), y_test)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    seen = X_train_orig
    y_seen = seen.TP
    X_seen = seen.loc[:,full_feat_set]
    X_seen.fillna(X_seen.mean(), inplace=True)  

    # balance train set with upsampling
    sm = RandomOverSampler(random_state=len_of_feat)
    X_seen, y_seen = sm.fit_resample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    unseen = X_test_heldout
    y_unseen = unseen.TP
    X_unseen = unseen.loc[:,full_feat_set]
    X_unseen.fillna(X_unseen.mean(), inplace=True) 

    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")


""
def calculate_ARIMA(tempdf, col_name, para_u, partial):
    
    start = dt.datetime.strptime("1 Nov 01", "%d %b %y")
    
    order = (1,1,0)
    
    if partial==1:
        temp_ts = tempdf.values[:, :-88]
    elif partial==0:
        temp_ts = tempdf.values[:, :-44]
    
    
    test = np.reshape(temp_ts, (temp_ts.shape[0], para_u, 44)) 
    test = np.nan_to_num(test,  neginf=0,  posinf=0 , nan = 0)
    

    daterange = pd.date_range(start, periods=para_u)
    ts_matrix = []
    for j in range(temp_ts.shape[0]):

        temp = []

        for i in range(44):

            ts_feats = test[j][:,i]
            ts_feats=  ts_feats.tolist()
            same = ts_feats.count(ts_feats[0]) == len(ts_feats)

            if not same:
                table = { "feat": ts_feats, "date": daterange}
                data = pd.DataFrame(table)
                data.set_index("date", inplace=True)
                #print(data)
                model = ARIMA(endog = data, order=order, freq='D')
                model = model.fit()
                prediction = model.predict(para_u, para_u).values[0]    
            else:
                prediction = ts_feats[0]
            temp.append(prediction)
        ts_matrix.append(temp)

        
    
    return ts_matrix


""
def top_final_partial(edges_orig, target_layer, predict_num,name, lstm): 
    
    try:
        os.mkdir("./feature_metrices/")
    except:
        print("")
    try:
        os.mkdir("./results/")
    except:
        print("")
    #if not os.path.isdir("./feature_metrices/"):
        #os.mkdir("./feature_metrices/")
    #if not os.path.isdir("./results/"):
        #os.mkdir("./results/")



    ts_col = ['ts_i', 'ts_j', 'ts_com_ne', 'ts_ave_deg_net', 'ts_var_deg_net', 'ts_ave_clust_net', 'ts_num_triangles_1', 
     'ts_num_triangles_2','ts_pag_rank1', 'ts_pag_rank2', 'ts_clust_coeff1', 'ts_clust_coeff2', 'ts_ave_neigh_deg1', 
     'ts_ave_neigh_deg2', 'ts_eig_cent1','ts_eig_cent2', 'ts_deg_cent1', 'ts_deg_cent2', 'ts_clos_cent1', 'ts_clos_cent2', 
     'ts_betw_cent1', 'ts_betw_cent2', 'ts_load_cent1', 'ts_load_cent2', 'ts_ktz_cent1', 'ts_ktz_cent2', 'ts_pref_attach', 
     'ts_LHN', 'ts_svd_edges', 'ts_svd_edges_dot', 'ts_svd_edges_mean', 'ts_svd_edges_approx', 'ts_svd_edges_dot_approx', 
     'ts_svd_edges_mean_approx', 'ts_short_path', 'ts_deg_assort', 'ts_transit_net', 'ts_diam_net', 'ts_num_nodes', 
     'ts_num_edges', 'ts_ind', 'ts_jacc_coeff', 'ts_res_alloc_ind', 'ts_adam_adar']


    colname = ['i', 'j', 'com_ne', 'ave_deg_net', 'var_deg_net', 'ave_clust_net', 'num_triangles_1', 'num_triangles_2', 'pag_rank1', 
     'pag_rank2', 'clust_coeff1', 'clust_coeff2', 'ave_neigh_deg1', 'ave_neigh_deg2', 'eig_cent1', 'eig_cent2', 'deg_cent1', 
     'deg_cent2', 'clos_cent1', 'clos_cent2', 'betw_cent1', 'betw_cent2', 'load_cent1', 'load_cent2', 'ktz_cent1', 'ktz_cent2', 
     'pref_attach', 'LHN', 'svd_edges', 'svd_edges_dot', 'svd_edges_mean', 'svd_edges_approx', 'svd_edges_dot_approx', 
     'svd_edges_mean_approx', 'short_path','deg_assort', 'transit_net', 'diam_net', 'num_nodes', 'num_edges', 'ind', 
     'jacc_coeff', 'res_alloc_ind', 'adam_adar']




    row_tr = []
    col_tr = []
    data_aux_tr = []
    A_tr = []

    edge_t_tr = []
    edge_f_tr = []
    df_f_tr = [] 
    df_t_tr = []


    #### load target layer A

    mymax= mxx(edges_orig)
    num_nodes = int(max( np.max(target_layer), mymax)) +1
    #num_nodes = int(np.max(target_layer)) + 1
    row = np.array(target_layer)[:,0]
    col = np.array(target_layer)[:,1]
    data_aux = np.ones(len(row))

    A = csr_matrix((data_aux,(row,col)),shape=(num_nodes,num_nodes))
    A = sparse.triu(A,1) + sparse.triu(A,1).transpose()
    A[A>0] = 1 
    A = A.todense()


    for i in range(len(edges_orig)):      
        row_tr.append(np.array(edges_orig[i])[:,0])
        col_tr.append(np.array(edges_orig[i])[:,1])
        data_aux_tr.append(np.ones(len(row_tr[i])))

        A_tr.append(csr_matrix((data_aux_tr[i],(row_tr[i],col_tr[i])),shape=(num_nodes,num_nodes)))
        A_tr[i] = sparse.triu(A_tr[i],1) + sparse.triu(A_tr[i],1).transpose()
        A_tr[i][A_tr[i]>0] = 1 
        A_tr[i] = A_tr[i].todense()        


    #### construct the holdout and training matriced from the original matrix
    alpha = 0.8 # sampling rate for holdout matrix
    alpha_ = 0.8 # sampling rate for training matrix

    A_ho = []
    A_tr_new = []


    for i in range(predict_num, len(A_tr)):
        A_hold, A_train = gen_tr_ho_networks(A_tr[i], alpha, alpha_)
        A_ho.append(A_hold)
        A_tr_new.append(A_train)

    ### here we are still missing the last training and testing label, we need to add the target layer
    A_hold_, A_train_ = gen_tr_ho_networks(A, alpha, alpha_)    
    A_tr_new.append(A_train_)
    ### A_ho is the test sets, the label is. The last element of A_ho is the true label where we try to predict. 
    A_ho.append(A_hold_)

    ### Now the training and hold out matrix list is complete, we try to create the corresponding edge lists for each.
    sample_true_false_edges_revised_5(A, A_ho, A_tr, A_tr_new, predict_num,name)

    for i in range(predict_num, len(A_tr)):
        temp_true_edges = np.loadtxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt").astype('int')
        edge_t_tr.append(temp_true_edges)

        temp_t_lstm = lstm[i-predict_num][temp_true_edges[:,0], temp_true_edges[:,1]]
        #lstm_t_feat.append(temp_t_lstm)

        temp_false_edges = np.loadtxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt").astype('int')       
        edge_f_tr.append(temp_false_edges)

        temp_f_lstm = lstm[i-predict_num][temp_false_edges[:,0], temp_false_edges[:,1]]
        #lstm_f_feat.append(temp_f_lstm)

        A_tr_temp = A_tr[i-predict_num:i]
        A_tr_temp.append(np.asmatrix(A_tr_new[i-predict_num]))


        #return A_tr[i], A_tr_temp, edge_f_tr[i-predict_num],A_tr[i], A_tr_temp, edge_f_tr[i-predict_num]

        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr_temp, edge_f_tr[i-predict_num])

        ### add the features of time series

        #tsm= calculate_ARIMA(df_temp1, ts_col, predict_num, 1)
        #df_temp2 = pd.DataFrame(tsm, columns = ts_col)


        #df_temp = pd.concat([df_temp1, df_temp2], axis=1)


        ### add the features of the lstm
        df_temp["lstm"] = temp_f_lstm

        df_f_tr.append(df_temp)

        ### do the same thing for the true edges 
        df_temp, time_temp = gen_topol_feats_3(A_tr[i], A_tr_temp, edge_t_tr[i-predict_num])



        #tsm= calculate_ARIMA(df_temp1, ts_col, predict_num, 1)
        #df_temp2 = pd.DataFrame(tsm, columns = ts_col)

        #df_temp = pd.concat([df_temp1, df_temp2], axis=1)


        df_temp["lstm"] = temp_t_lstm        
        df_t_tr.append(df_temp)
        
    edge_t_tr_final= np.loadtxt("./edge_tf_tr/edge_t_final"+"_"+str(name)+".txt").astype('int')
    edge_f_tr_final = np.loadtxt("./edge_tf_tr/edge_f_final"+"_"+str(name)+".txt").astype('int')
    A_tr_temp_ = A_tr[-(predict_num):]
    A_tr_temp_.append(np.asmatrix(A_train_))
    df_f_temp, time1 = gen_topol_feats_3(A_hold_, A_tr_temp_, edge_f_tr_final)
    df_t_temp, time2 = gen_topol_feats_3(A_hold_, A_tr_temp_, edge_t_tr_final)


    #tsm_f= calculate_ARIMA(df_f_temp_1, ts_col, predict_num, 1)
    #tsm_t = calculate_ARIMA(df_t_temp_1, ts_col, predict_num, 1)
    #df_tempf = pd.DataFrame(tsm_f, columns = ts_col)
    #df_tempt = pd.DataFrame(tsm_t, columns = ts_col)

    #df_f_temp = pd.concat([df_f_temp1, df_tempf], axis=1)
    #df_t_temp = pd.concat([df_t_temp1, df_tempt], axis=1)



    temp_t_lstm = lstm[-1][edge_t_tr_final[:,0], edge_t_tr_final[:,1]]
    temp_f_lstm = lstm[-1][edge_f_tr_final[:,0], edge_f_tr_final[:,1]]

    df_t_temp["lstm"] = temp_t_lstm
    df_f_temp["lstm"] = temp_f_lstm

    df_f_tr.append(df_f_temp)
    df_t_tr.append(df_t_temp)


    edge_t_ho= np.loadtxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt").astype('int')
    edge_f_ho= np.loadtxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt").astype('int')

    A_tr_temp_ = A_tr[-(predict_num):]
    A_tr_temp_.append(np.asmatrix(A_train_))
    df_f_ho, time1 = gen_topol_feats_3(A, A_tr_temp_, edge_f_ho)
    df_t_ho, time2 = gen_topol_feats_3(A, A_tr_temp_, edge_t_ho)


    #tsm_f= calculate_ARIMA(df_f_ho_1, ts_col, predict_num, 1)
    #tsm_t = calculate_ARIMA(df_t_ho_1, ts_col, predict_num, 1)
    #df_tempf = pd.DataFrame(tsm_f, columns = ts_col)
    #df_tempt = pd.DataFrame(tsm_t, columns = ts_col)

    #df_f_ho = pd.concat([df_f_ho1, df_tempf], axis=1)
    #df_t_ho = pd.concat([df_t_ho1, df_tempt], axis=1)





    temp_t_lstm = lstm[-1][edge_t_ho[:,0], edge_t_ho[:,1]]
    temp_f_lstm = lstm[-1][edge_f_ho[:,0], edge_f_ho[:,1]]

    df_t_ho["lstm"] = temp_t_lstm
    df_f_ho["lstm"] = temp_f_lstm
    
    
    
    
    train = []

    for i in range(len(A_tr)):
        train.append(np.loadtxt("./for_sbm/temp_train_{}".format(i)+"_"+str(name)+".txt").astype('int'))


    for i in range(predict_num, len(A_tr)):
        dummy = train[i-predict_num]
        dummy[:, 2] =  0 
        count = 1
        for j in range(i-predict_num+1, i+1):
            temp_dummy = train[j]
            temp_dummy[:,2] = count
            count = count+1
            dummy = np.vstack((dummy, temp_dummy))    
        np.savetxt("./for_sbm/train_{}".format(i)+"_"+str(name)+".txt", dummy, fmt='%u')



    dummy1 = train[len(A_tr)-predict_num]
    dummy1[:, 2] =  0
    count = 1
    for j in range(len(A_tr)-predict_num+1, len(A_tr)):
        temp_dummy1 = train[j]
        temp_dummy1[:,2] = count
        count = count+1
        dummy1 = np.vstack((dummy1, temp_dummy1))

    dummy2 = np.loadtxt("./for_sbm/train_final"+"_"+str(name)+".txt")    
    dummy2[:,2] = count
    dummy1 = np.vstack((dummy1, dummy2))
    np.savetxt("./for_sbm/train_last_layer"+"_"+str(name)+".txt", dummy1, fmt='%u')


    sbm_output = []
    for i in range(predict_num, len(A_tr)):
        train = "./for_sbm/train_{}".format(i)+"_"+str(name)+".txt"
        test = "./for_sbm/test_{}".format(i)+"_"+str(name)+".txt"
        sbm_output.append(sbm_model(train, test, num_nodes, predict_num+1))


    sbm_final = sbm_model("./for_sbm/train_last_layer"+"_"+str(name)+".txt",
                          "./for_sbm/test_final"+"_"+str(name)+".txt", num_nodes, predict_num+1)


    sbm_final = np.array(sbm_final)
    sbm_final_true = sbm_final[:,1][0::2]
    sbm_final_false = sbm_final[:,1][1::2]

    sbm_true = []
    sbm_false = []

    for item in sbm_output:
        sbm_m =  np.array(item)
        sbm_true.append(sbm_m[:,1][0::2])
        sbm_false.append(sbm_m[:,1][1::2])


    for i in range(len(sbm_true)):
        df_t_tr[i]["sbm"] = sbm_true[i]
        df_f_tr[i]["sbm"] = sbm_false[i]


    df_t_tr_columns = df_t_tr[0].columns
    df_f_tr_columns = df_f_tr[0].columns    

    v1 = df_t_tr[0].values
    v2 = df_f_tr[0].values


    # Stack Together for df_f_ho, df_f_tr, df_t_ho, df_t_tr
    # Let the randomforest make selection

    for i in range(1, len(sbm_true)):
        v1 = np.vstack((v1,df_t_tr[i].values))
        v2 = np.vstack((v2,df_f_tr[i].values))

    #   return df_t_tr, df_f_tr, df_t_ho, df_f_ho  
    df_t_tr_ = pd.DataFrame(data=v1, columns = df_t_tr_columns)
    df_f_tr_ = pd.DataFrame(data=v2, columns = df_f_tr_columns)

    column_name = list(df_t_tr_.columns)

    for j in range(1,predict_num+2):  
        for i in range (44*j, 44*(j+1)):
            column_name[i] = column_name[i]+"_"+str(j)


    df_t_ho["sbm"] = sbm_final_true
    df_f_ho["sbm"] = sbm_final_false


    df_t_tr_.columns = column_name
    df_f_tr_.columns = column_name
    df_t_ho.columns = column_name
    df_f_ho.columns = column_name
    
    
    feat_path = "./ef_gen_tr/"
    try:
        os.mkdir(feat_path)
    except:
        print("")
    #if not os.path.isdir(feat_path):
        #os.mkdir(feat_path)
    df_t_tr_.to_pickle(feat_path + 'df_t'+"_"+str(name))
    df_f_tr_.to_pickle(feat_path + 'df_f'+"_"+str(name))

    feat_path = "./ef_gen_ho/"
    try:
        os.mkdir(feat_path)
    except:
        print("")
    #if not os.path.isdir(feat_path):
        #os.mkdir(feat_path)
    df_t_ho.to_pickle(feat_path + 'df_t'+"_"+str(name))
    df_f_ho.to_pickle(feat_path + 'df_f'+"_"+str(name))

