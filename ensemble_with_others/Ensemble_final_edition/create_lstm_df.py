import numpy as np 
import pandas as pd
import pickle
from multiprocessing import Pool

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


def generate_lstm_df(name):

    train_lstm = np.load("./lstm_feat//"  +  name +"_train.npy")
    test_lstm = np.load("./lstm_feat//"  +  name +"_test.npy")     

    train_lstm = np.array(train_lstm)
    test_lstm = np.array(test_lstm)

    predict_num = 3
    train_length = 6

    lstm_col = ["lstm1", "lstm2", "lstm3", "lstm4", "lstm5", "lstm6"]


    feat_path = "./all_features/"

    with open(feat_path + 'df_t_tr'+"_"+str(name), 'rb') as pk:
        df_t_tr_ts = pickle.load(pk)

    with open(feat_path + 'df_f_tr'+"_"+str(name), 'rb') as pk:
        df_f_tr_ts = pickle.load(pk)


    with open(feat_path + 'df_t_ho'+"_"+str(name), 'rb') as pk:
        df_t_ho_ts = pickle.load(pk)

    with open(feat_path + 'df_f_ho'+"_"+str(name), 'rb') as pk:
        df_f_ho_ts = pickle.load(pk)





    init_t = np.empty([6,1])
    init_f = np.empty([6,1])


    for i in range(predict_num, train_length):


        temp_true_edges = np.loadtxt("./edge_tf_tr/edge_t_{}".format(i)+"_"+str(name)+".txt").astype('int')        
        temp_false_edges = np.loadtxt("./edge_tf_tr/edge_f_{}".format(i)+"_"+str(name)+".txt").astype('int')       




        t_temp = []
        f_temp = []

        for i in range(3):

            t_temp.append(train_lstm[i][temp_true_edges[:,0]])
            t_temp.append(train_lstm[i][temp_true_edges[:,1]])


            f_temp.append(train_lstm[i][temp_false_edges[:,0]])
            f_temp.append(train_lstm[i][temp_false_edges[:,1]])


        t_temp = np.array(t_temp)    
        f_temp = np.array(f_temp)


        init_t = np.hstack((init_t,t_temp))
        init_f = np.hstack((init_f,f_temp))


    edge_t_tr_final= np.loadtxt("./edge_tf_tr/edge_t_final"+"_"+str(name)+".txt").astype('int')
    edge_f_tr_final = np.loadtxt("./edge_tf_tr/edge_f_final"+"_"+str(name)+".txt").astype('int')



    #temp_t_lstm = train_lstm[edge_t_tr_final[:,0], edge_t_tr_final[:,1]]
    #temp_f_lstm = test_lstm[edge_f_tr_final[:,0], edge_f_tr_final[:,1]]

    #df_t_temp["lstm"] = temp_t_lstm
    #df_f_temp["lstm"] = temp_f_lstm


    t_temp = []
    f_temp = []

    for i in range(3):


        t_temp.append(train_lstm[i][edge_t_tr_final[:,0]])
        t_temp.append(train_lstm[i][edge_t_tr_final[:,1]])


        f_temp.append(train_lstm[i][edge_f_tr_final[:,0]])
        f_temp.append(train_lstm[i][edge_f_tr_final[:,1]])


    t_temp = np.array(t_temp)    
    f_temp = np.array(f_temp)

    lstm_tr_t = np.hstack((init_t, t_temp))
    lstm_tr_f = np.hstack((init_f, f_temp))


    lstm_tr_t = lstm_tr_t.T[1:]
    lstm_tr_f = lstm_tr_f.T[1:]


    df_temp_tr_t = pd.DataFrame(lstm_tr_t, columns = lstm_col)
    df_temp_tr_f = pd.DataFrame(lstm_tr_f, columns = lstm_col)

    df_f_tr = pd.concat([df_f_tr_ts, df_temp_tr_f], axis=1)
    df_t_tr = pd.concat([df_t_tr_ts, df_temp_tr_t], axis=1)

    edge_t_ho= np.loadtxt("./edge_tf_true/edge_t"+"_"+str(name)+".txt").astype('int')
    edge_f_ho= np.loadtxt("./edge_tf_true/edge_f"+"_"+str(name)+".txt").astype('int')



    t_temp = []
    f_temp = []

    for i in range(3):


        t_temp.append(test_lstm[i][edge_t_ho[:,0]])
        t_temp.append(test_lstm[i][edge_t_ho[:,1]])

        f_temp.append(test_lstm[i][edge_f_ho[:,0]])
        f_temp.append(test_lstm[i][edge_f_ho[:,1]])


    lstm_ho_t = np.array(t_temp)    
    lstm_ho_f = np.array(f_temp)

    df_temp_ho_t = pd.DataFrame(lstm_ho_t.T, columns = lstm_col)
    df_temp_ho_f = pd.DataFrame(lstm_ho_f.T, columns = lstm_col)

    df_f_ho = pd.concat([df_f_ho_ts, df_temp_ho_f], axis=1)
    df_t_ho = pd.concat([df_t_ho_ts, df_temp_ho_t], axis=1)

    if not os.path.isdir("./finalized_all_features/"):
        os.mkdir("./finalized_all_features/")

    feat_path = "./finalized_all_features/"
    df_t_tr.to_pickle(feat_path + 'df_t_tr'+"_"+str(name))
    df_f_tr.to_pickle(feat_path + 'df_f_tr'+"_"+str(name))
    df_t_ho.to_pickle(feat_path + 'df_t_ho'+"_"+str(name))
    df_f_ho.to_pickle(feat_path + 'df_f_ho'+"_"+str(name))
    
    
#with Pool(len(data_list)) as p:
#    print(p.map(generate_lstm_df, data_list))

data_name = "fake110"
generate_lstm_df(data_name)
