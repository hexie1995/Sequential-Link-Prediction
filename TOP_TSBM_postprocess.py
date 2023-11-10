# NOTE: This is intended as a help function for people who only wish to run T-SBM and Top without Time Series or E-LSTM-D. 
# There might be more work needed to be done after call this function. 

##### WARNINGS ######
# DO NOT RUN THIS FUNCTION IF YOU WANT TO RUN THE FULL ENSEMBLE-SEQUENTIAL method
# ONLY RUN WHEN YOU KNOW THAT AFTERWARDS YOU NEED TO DELETE THE FILES AND FOLDER "finalized_all_features" if you wish to continue run the whole method.
# PUT THIS IN THE SAME FOLDER THAT YOU ARE RUNNING THE FILE. 
##### WARNINGS ######

def post_process(name):

    if not os.path.isdir("./finalized_all_features/"):
        os.mkdir("./finalized_all_features/")
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
    
    

    feat_path = "./finalized_all_features/"
    df_t_tr.to_pickle(feat_path + 'df_t_tr'+"_"+str(name))
    df_f_tr.to_pickle(feat_path + 'df_f_tr'+"_"+str(name))
    df_t_ho.to_pickle(feat_path + 'df_t_ho'+"_"+str(name))
    df_f_ho.to_pickle(feat_path + 'df_f_ho'+"_"+str(name))

data_name = "fake110"
post_process(data_name)
