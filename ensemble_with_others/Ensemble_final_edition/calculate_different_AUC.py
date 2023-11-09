# +
import numpy as np
from multiprocessing import Pool
from TOLP import *
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
import pandas as pd
import warnings
import pickle

path = r"../../community_label_TSBM//"

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

data_list = ["chess","obrazil","bionet1", "bitcoin","emaildnc","bionet2",
              "obitcoin","london","collegemsg","fbmsg","radoslaw", "fbforum", "mit",
             "ant1","ant2","ant3","ant4","ant5","ant6"]




# -

def creat_numpy_files_predictor_choice(dir_results, name, predict, choice):

   
    feat_path = "./finalized_all_features/"
    
    with open(feat_path + 'df_t_tr'+"_"+str(name), 'rb') as pk:
        df_t_tr_ts = pickle.load(pk)

    with open(feat_path + 'df_f_tr'+"_"+str(name), 'rb') as pk:
        df_f_tr_ts = pickle.load(pk)


    with open(feat_path + 'df_t_ho'+"_"+str(name), 'rb') as pk:
        df_t_ho_ts = pickle.load(pk)

    with open(feat_path + 'df_f_ho'+"_"+str(name), 'rb') as pk:
        df_f_ho_ts = pickle.load(pk)



        

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
        full_feat_set = ["i", "j", "lstm1", "lstm2", "lstm3", "lstm4", "lstm5", "lstm6"]
    
    elif choice ==4:
        full_feat_set = base_top_set + ts_col + ["sbm"] +  ["lstm1", "lstm2", "lstm3", "lstm4", "lstm5", "lstm6"]

    
    
    X_train = pd.concat([df_t_tr_ts, df_f_tr_ts])
    y_train_orig = pd.DataFrame(np.array([1]*df_t_tr_ts.shape[0] + [0]*df_f_tr_ts.shape[0]), columns = ["TP"])
    

    X_test = pd.concat([df_t_ho_ts, df_f_ho_ts])
    y_test_orig =  pd.DataFrame(np.array([1]*df_t_ho_ts.shape[0] + [0]*df_f_ho_ts.shape[0]), columns = ["TP"])
    
    
    
    X_train_orig = X_train[full_feat_set]
    X_test_orig = X_test[full_feat_set]

    
    X_train_orig = X_train_orig.replace([np.inf, -np.inf, np.nan], 0)
    X_test_orig = X_test_orig.replace([np.inf, -np.inf, np.nan], 0)

    X_test_orig.fillna(X_test_orig.mean())
    X_train_orig.fillna(X_train_orig.mean())        

    print(X_test_orig.isnull().any().any())
    print(X_train_orig.isnull().any().any())
    
    #X_train_orig_val = np.nan_to_num(X_train_orig, nan = X_train_orig.mean(), posinf=0, neginf=0)
    #X_test_orig_val = np.nan_to_num(X_test_orig, nan = X_test_orig.mean(), posinf=0, neginf=0)

    #X_train_orig = pd.DataFrame(X_train_orig_val, columns = colmn)
    #X_test_orig = pd.DataFrame(X_test_orig_val, columns = colmn)
    
    
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    skf.get_n_splits(X_train_orig, y_train_orig)

    if not os.path.isdir(dir_results+'/'):
        os.mkdir(dir_results+'/')
        
    nFold = 1 
    for train_index, test_index in skf.split(X_train_orig, y_train_orig):

        cv_train = list(train_index)
        cv_test = list(test_index)
         
        Xtrain = X_train_orig.iloc[np.array(cv_train)]
        Xtest = X_train_orig.iloc[np.array(cv_test)]

        ytrain = y_train_orig.iloc[np.array(cv_train)]
        ytest = y_train_orig.iloc[np.array(cv_test)]
        
        
        #print(Xtest.shape)
        #print(ytest.shape)
        
        Xtest.fillna(Xtest.mean(), inplace=True)
        Xtrain.fillna(Xtrain.mean(), inplace=True)
        
        
        
        
        
        sm = RandomOverSampler(random_state=len_of_feat)
        Xtrain, ytrain = sm.fit_resample(Xtrain, ytrain)

        np.save(dir_results+'/X_trainE_'+'cv'+str(nFold), Xtrain)
        np.save(dir_results+'/y_trainE_'+'cv'+str(nFold), ytrain)
        np.save(dir_results+'/X_testE_'+'cv'+str(nFold), Xtest)
        np.save(dir_results+'/y_testE_'+'cv'+str(nFold), ytest)

        print( "created fold ",nFold, " ...")
        
        nFold = nFold + 1

    X_seen = X_train_orig
    y_seen = y_train_orig

    #X_seen.fillna(X_seen.mean(), inplace=True)  
    # balance train set with upsampling
    sm = RandomOverSampler(random_state=len_of_feat)
    X_seen, y_seen = sm.fit_resample(X_seen, y_seen)

    np.save(dir_results+'/X_Eseen', X_seen)
    np.save(dir_results+'/y_Eseen', y_seen)
    print( "created train set ...")


    X_unseen = X_test_orig
    y_unseen = y_test_orig
    #X_unseen.fillna(X_unseen.mean(), inplace=True)  
    
    np.save(dir_results+'/X_Eunseen', X_unseen)
    np.save(dir_results+'/y_Eunseen', y_unseen) 
    print( "created holdout set ...")


# +
def produce_AUC(name, predict_num, choice):
    
    dir_output = "./feature_metrices" + "/"
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)
    
    #### creat and save feature matrices #### 
    dir_output = './feature_metrices'+"/"+str(name) +"/" # output path
    if not os.path.isdir(dir_output):
        os.mkdir(dir_output)

    creat_numpy_files_predictor_choice(dir_output, name, predict_num+1, choice)

    path_to_data = './feature_metrices' +"/"+str(name)
    path_to_results = './results'+"/"+str(name)
    n_depths = [3, 6] # here is a sample search space
    n_ests = [25, 50, 100] # here is a sample search space
    
    n_depth, n_est, var = model_selection(path_to_data, path_to_results, n_depths, n_ests)

    
    auprc, auc, precision,recall, featim = heldout_performance_bestchoice(path_to_data, path_to_results, n_depth, n_est)
    
    return auprc, auc, precision,recall, featim, var
    
    


def AUC_wrapper(name):
    
    AUPRC = []
    AUC = []
    PRE = []
    REC = []
    FEATIM = []
    VAR = []
    
    predict_num = 3
    for choice in range(5):
        
        auprc, auc, precision,recall, featim, var = produce_AUC(name, predict_num, choice)   
        AUPRC.append(auprc)
        AUC.append(auc)
        PRE.append(precision)
        REC.append(recall)
        FEATIM.append(featim)
        VAR.append(var)
 
    np.save("full_results_final/AUPRC_"+(name)+".npy",AUPRC)
    np.save("full_results_final/AUC_"+(name)+".npy",AUC)
    np.save("full_results_final/PRE_"+(name)+".npy",PRE)
    np.save("full_results_final/REC_"+(name)+".npy",REC)
    np.save("full_results_final/FEATIM_"+(name)+".npy",FEATIM)
    np.save("full_results_final/VAR_"+(name)+".npy",VAR)



#with Pool(len(data_list)) as p:
#    print(p.map(AUC_wrapper, data_list))


data_name = "fake110"
run_data(data_name)
