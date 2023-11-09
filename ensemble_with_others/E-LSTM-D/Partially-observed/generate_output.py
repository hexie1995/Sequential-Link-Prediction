import os
from multiprocessing import Pool
import numpy as np

path = r"../../../community_label_TSBM//"

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
    if not os.path.isdir("./lstm_feat/"):
	    os.mkdir("./lstm_feat/")
    
    trainX = np.load("./matrix//"  +  name +"_train.npy")
    testX = np.load("./matrix//"  +  name +"_test.npy")     
    weights = np.load("./weights//"  +  name +".npy",allow_pickle=True)

    train_feat = []
    test_feat = []
    temp = np.matmul(weights[0],weights[2])
    temp = np.matmul(temp, weights[3].T)
    wmat = np.matmul(temp, weights[8])



    for i in range(trainX.shape[1]):
        result = np.matmul(trainX[0][i], wmat)
        train_feat.append(np.sum(result, axis = 1))
        result = np.matmul(testX[0][i], wmat)
        test_feat.append(np.sum(result, axis = 1))

    np.save("./lstm_feat//"  +  name +"_train.npy", train_feat)
    np.save("./lstm_feat//"  +  name +"_test.npy", test_feat)     

#with Pool(len(data_list)) as p:
#    print(p.map(save_feat_matrix, data_list))

data_name = "fake110"

save_feat_matrix(data_name)
