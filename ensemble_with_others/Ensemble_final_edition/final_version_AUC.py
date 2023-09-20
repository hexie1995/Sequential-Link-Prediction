# +
import numpy as np
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import datetime as dt
import pandas as pd
import warnings
import pickle


warnings.filterwarnings("ignore")

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

data_list1 = ["chess","obrazil","bionet1", "bitcoin","emaildnc","bionet2",
              "obitcoin","london","collegemsg","fbmsg","radoslaw", "fbforum", "mit",
             "ant1","ant2","ant3","ant4","ant5","ant6"]




# -

def rf_with_chosen_feats(df_ho,df_tr, predict, choice):
    
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
    

    ts_col = ['ts_i', 'ts_j', 'ts_com_ne', 'ts_ave_deg_net', 'ts_var_deg_net', 'ts_ave_clust_net', 'ts_num_triangles_1', 
     'ts_num_triangles_2','ts_pag_rank1', 'ts_pag_rank2', 'ts_clust_coeff1', 'ts_clust_coeff2', 'ts_ave_neigh_deg1', 
     'ts_ave_neigh_deg2', 'ts_eig_cent1','ts_eig_cent2', 'ts_deg_cent1', 'ts_deg_cent2', 'ts_clos_cent1', 'ts_clos_cent2', 
     'ts_betw_cent1', 'ts_betw_cent2', 'ts_load_cent1', 'ts_load_cent2', 'ts_ktz_cent1', 'ts_ktz_cent2', 'ts_pref_attach', 
     'ts_LHN', 'ts_svd_edges', 'ts_svd_edges_dot', 'ts_svd_edges_mean', 'ts_svd_edges_approx', 'ts_svd_edges_dot_approx', 
     'ts_svd_edges_mean_approx', 'ts_short_path', 'ts_deg_assort', 'ts_transit_net', 'ts_diam_net', 'ts_num_nodes', 
     'ts_num_edges', 'ts_ind', 'ts_jacc_coeff', 'ts_res_alloc_ind', 'ts_adam_adar']

   
    base_top_set = [None] * ((predict)*len(feature_set))
    len_of_feat = len(feature_set)
    
    for k in range(len_of_feat):
        base_top_set[k] = feature_set[k]
    
    for j in range(1,predict):  
        for i in range (len_of_feat*j, len_of_feat*(j+1)):
            base_top_set[i] = feature_set[i-len_of_feat*j]+"_"+str(j)
        
    #print(full_feat_set)
    
    
    if choice == 0 :
        
        full_feat_set = base_top_set
    elif choice ==1:
        
        full_feat_set = ts_col
    elif choice ==2:
        
        full_feat_set = ["i", "j" , "sbm"]
    
    elif choice ==3:
        full_feat_set = ["i", "j", "lstm"]
    
    elif choice ==4:
        full_feat_set = base_top_set + ts_col + ["sbm", "lstm"] 

        
        
        
    
    X_test_ts = df_ho
    y_test_ts = np.array(df_ho.TP)
    
    
    X_train_ts = df_tr
    y_train_ts = np.array(df_tr.TP)
    
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    col_mean = np.nanmean(X_train_ts, axis=0)
    X_train_ts = np.nan_to_num(X_train_ts, nan=0, posinf=0, neginf=0)


    col_mean = np.nanmean(X_test_ts, axis=0)
    X_test_ts = np.nan_to_num(X_test_ts, nan=0, posinf=0, neginf=0)
    if chosen_feat!=100:
        dtree_model = RandomForestClassifier(max_depth=2).fit(X_train_ts.reshape(-1, 1), y_train_ts)

    elif chosen_feat==100:
        dtree_model = RandomForestClassifier(max_depth=2).fit(X_train_ts, y_train_ts)

        
    # feature importance and prediction on test set 
    #feature_importance = dtree_model.feature_importances_
    
    if chosen_feat!=100:
    
        dtree_predictions = dtree_model.predict(X_test_ts.reshape(-1, 1))
        dtree_proba = dtree_model.predict_proba(X_test_ts.reshape(-1, 1))
    elif chosen_feat==100:
        dtree_predictions = dtree_model.predict(X_test_ts)
        dtree_proba = dtree_model.predict_proba(X_test_ts)
         

    cm_dt4 = confusion_matrix(y_test_ts, dtree_predictions)
    auc_measure = roc_auc_score(y_test_ts, dtree_proba[:,1])
    auprc = average_precision_score(y_test_ts, dtree_proba[:,1])
    precision_total, recall_total, f_measure_total, _ = precision_recall_fscore_support(y_test_ts, dtree_predictions, average=None)

    return auprc, auc_measure, precision_total, recall_total, cm_dt4


# +
def produce_AUC(name, predict_num, choice):
    feat_path = "./all_features/"
    
    with open(feat_path + 'df_t_tr'+"_"+str(name), 'rb') as pk:
        df_t_tr_ = pickle.load(pk)

    with open(feat_path + 'df_f_tr'+"_"+str(name), 'rb') as pk:
        df_f_tr_ = pickle.load(pk)


    with open(feat_path + 'df_t_ho'+"_"+str(name), 'rb') as pk:
        df_t_ho = pickle.load(pk)

    with open(feat_path + 'df_f_ho'+"_"+str(name), 'rb') as pk:
        df_f_ho = pickle.load(pk)
    
    df_tr = creat_full_set_temporal(df_t_tr_, df_f_tr_, predict_num+2)
    df_ho = creat_full_set_temporal(df_t_ho, df_f_ho, predict_num+2)

    dir_output = "./feature_metrices" + "/"
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'+"/"+str(name) +"/" # output path
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    # here we only +1 , because we are ignoring the last layer of information. which is just the original matrix.
    # the plus one is plusing towards the length of the feature vectors. 
    auprc, auc_measure, precision, recall, cm_dt4 = rf_with_chosen_feats(df_ho, df_tr, predict_num+1, choice)

    return auprc, auc, precision,recall, featim, cm_dt
    
    


def AUC_wrapper(name):
    
    AUPRC = []
    AUC = []
    PRE = []
    REC = []
    CM = []
    
    predict_num = 3
    for choice in range(5):
        
        auprc, auc, precision,recall, featim, cmdt4 = produce_AUC(name, predict_num, choice)   
        AUPRC.append(auprc)
        AUC.append(auc)
        PRE.append(precision)
        REC.append(recall)
        CM.append(cmdt4)

    np.save("fake1_final_partial_results/AUPRC_"+(name)+".npy",AUPRC)
    np.save("fake1_final_partial_results/AUC_"+(name)+".npy",AUC)
    np.save("fake1_final_partial_results/PRE_"+(name)+".npy",PRE)
    np.save("fake1_final_partial_results/REC_"+(name)+".npy",REC)
    np.save("fake1_final_partial_results/CM_"+(name)+".npy",CM)    
    



with Pool(len(fakelist)) as p:
    print(p.map(save_df_ts, fakelist))


#AUC_wrapper(fakelist[0])
