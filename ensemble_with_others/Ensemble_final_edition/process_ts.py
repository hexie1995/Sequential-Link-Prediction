import numpy as np
#from TOLPIND import *
from multiprocessing import Pool
#import TOLPTS as tolp
import OLP_FINAL_PARTIAL as tolp
#from statsforecast import StatsForecast
#from statsforecast.models import AutoARIMA
from statsmodels.tsa.arima.model import ARIMA
path = r"../../community_label_TSBM//"
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



def calculate_ARIMA(tempdf, para, partial):
    
    start = dt.datetime.strptime("1 Nov 01", "%d %b %y")
    
    order = (0,2,0)
    
    if partial==1:
        para = para + 1
    elif partial==0:
        pass
        
    temp_ts = tempdf.values[:, :-46]
    test = np.reshape(temp_ts, (temp_ts.shape[0], para, 44)) 
    test = np.nan_to_num(test,  neginf=0,  posinf=0 , nan = 0)
    

    daterange = pd.date_range(start, periods=para)
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
                try:
                
                    model = ARIMA(endog = data, order=order, freq='D')
                    model = model.fit()
                    prediction = model.predict(para, para).values[0]    
                except:
                    prediction = ts_feats[0]
            else:
                prediction = ts_feats[0]
            temp.append(prediction)
        ts_matrix.append(temp)
    
    return ts_matrix



ts_col = ['ts_i', 'ts_j', 'ts_com_ne', 'ts_ave_deg_net', 'ts_var_deg_net', 'ts_ave_clust_net', 'ts_num_triangles_1', 
 'ts_num_triangles_2','ts_pag_rank1', 'ts_pag_rank2', 'ts_clust_coeff1', 'ts_clust_coeff2', 'ts_ave_neigh_deg1', 
 'ts_ave_neigh_deg2', 'ts_eig_cent1','ts_eig_cent2', 'ts_deg_cent1', 'ts_deg_cent2', 'ts_clos_cent1', 'ts_clos_cent2', 
 'ts_betw_cent1', 'ts_betw_cent2', 'ts_load_cent1', 'ts_load_cent2', 'ts_ktz_cent1', 'ts_ktz_cent2', 'ts_pref_attach', 
 'ts_LHN', 'ts_svd_edges', 'ts_svd_edges_dot', 'ts_svd_edges_mean', 'ts_svd_edges_approx', 'ts_svd_edges_dot_approx', 
 'ts_svd_edges_mean_approx', 'ts_short_path', 'ts_deg_assort', 'ts_transit_net', 'ts_diam_net', 'ts_num_nodes', 
 'ts_num_edges', 'ts_ind', 'ts_jacc_coeff', 'ts_res_alloc_ind', 'ts_adam_adar']


def save_df_ts(name):
    
    predict_num = 3
    warnings.filterwarnings("ignore")
    feat_path = "./ef_gen_tr/"

    with open(feat_path + 'df_t'+"_"+str(name), 'rb') as pk:
        df_t_tr = pickle.load(pk)

    with open(feat_path + 'df_f'+"_"+str(name), 'rb') as pk:
        df_f_tr = pickle.load(pk)

    feat_path = "./ef_gen_ho/"


    with open(feat_path + 'df_t'+"_"+str(name), 'rb') as pk:
        df_t_ho = pickle.load(pk)

    with open(feat_path + 'df_f'+"_"+str(name), 'rb') as pk:
        df_f_ho = pickle.load(pk)
    
    para_u = predict_num
    
    
    t_tr_mat = calculate_ARIMA(df_t_tr, para_u, 1)
    t_tr_df =  pd.DataFrame(t_tr_mat, columns = ts_col)
    df_t_tr_ = pd.concat([df_t_tr, t_tr_df], axis=1)

    feat_path = "./all_features/"
    df_t_tr_.to_pickle(feat_path + 'df_t_tr'+"_"+str(name))

    
    f_tr_mat = calculate_ARIMA(df_f_tr, para_u, 1)
    f_tr_df =  pd.DataFrame(f_tr_mat, columns = ts_col)
    df_f_tr_ = pd.concat([df_f_tr, f_tr_df], axis=1)

    feat_path = "./all_features/"
    df_f_tr_.to_pickle(feat_path + 'df_f_tr'+"_"+str(name))

    
    
    t_ho_mat = calculate_ARIMA(df_t_ho, para_u, 1)
    t_ho_df =  pd.DataFrame(t_ho_mat, columns = ts_col)
    df_t_ho_ = pd.concat([df_t_ho, t_ho_df], axis=1)

    feat_path = "./all_features/"
    df_t_ho_.to_pickle(feat_path + 'df_t_ho'+"_"+str(name))

    
    f_ho_mat = calculate_ARIMA(df_f_ho, para_u, 1)
    f_ho_df =  pd.DataFrame(f_ho_mat, columns = ts_col)
    df_f_ho_ = pd.concat([df_f_ho, f_ho_df], axis=1)

    feat_path = "./all_features/"
    df_f_ho_.to_pickle(feat_path + 'df_f_ho'+"_"+str(name))


#with Pool(len(data_list1)) as p:
#    print(p.map(save_df_ts, data_list1))

data_name = "fake110"
save_df_ts(data_name)
