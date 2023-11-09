from model import e_lstm_d
import os
from multiprocessing import Pool
import tensorflow as tf
import numpy as np
from utils import *
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



def save_feat_matrix(name):
    hist_len = 3
    #hist_len = hist_len - 1
    encoder = [128]
    lstm = [256,256]
 
    
    data = load_data(path+name+"/"+name+"_partial.npy")
    model = e_lstm_d(num_nodes=data.shape[1], historical_len=hist_len, encoder_units=[int(x) for x in encoder], 
                     lstm_units=[int(x) for x in lstm], 
                     decoder_units=[int(x) for x in [data.shape[1]]],
                     name=name)

    weights = model.model.get_weights()
    
    np.save("./weights//"  +  name +".npy", weights)
    
    trainX = np.array([data[k+3: hist_len +k+3] for k in range(1)], dtype=np.float32)
    trainY = np.array(data[6:7], dtype=np.float32)
    testX = np.array([data[4:7]], dtype=np.float32)
    testY = np.array(data[7:8], dtype=np.float32)
    
    np.save("./matrix//"  +  name +"_train.npy", trainX)
    np.save("./matrix//"  +  name +"_test.npy", testX) 
    
    history = model.train(trainX, trainY)
    aucs, err_rates, y_pred, y = model.evaluate(testX, testY)  
    
    
    np.save("./ys//"  +  name +"_ypred.npy", y_pred)
    np.save("./ys//"  +  name +"_ytrue.npy", y)     
    np.save("./auc//" + name +".npy", auc)
    
data_name = "fake110"   

save_feat_matrix(path+data_name)

#with Pool(len(data_list)) as p:
#    print(p.map(save_feat_matrix, data_list))
