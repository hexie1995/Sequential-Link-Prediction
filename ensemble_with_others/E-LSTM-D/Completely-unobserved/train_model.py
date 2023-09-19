# -*- coding: utf-8 -*-
from model import e_lstm_d
import os
import tensorflow as tf
import numpy as np
from utils import *
import tensorflow.keras.backend as K


path = r"C:/Users/hexie/Desktop/Multilayer_Link/real_data/real_data//"
data_list1 = ["bionet1","bionet2","chess", "bitcoin","collegemsg","obitcoin","obrazil","radoslaw", "london", "mit"]
data_list2 = ["ant1","ant2","ant3","ant4","ant5","ant6"]
data_list3 = ["fbforum","fbmsg","emaildnc"]
data_list = data_list2 + data_list3

# +
#flags = tf.compat.v1.flags
#FLAGS = flags.FLAGS
#flags.DEFINE_string('GPU', '0', 'train model on which GPU devide. -1 for CPU')
#flags.DEFINE_string('dataset', 'contact', 'the dataset used for training and testing')
#flags.DEFINE_integer('historical_len', 3, 'number of historial snapshots each sample')
#flags.DEFINE_string('encoder', [int(128)] , 'encoder structure parameters')
#flags.DEFINE_string('lstm', [256,256], 'stacked lstm structure parameters')
#flags.DEFINE_string('decoder',[274], 'decoder structure parameters')
#flags.DEFINE_integer('num_epochs', 800, 'Number of training epochs.')
#flags.DEFINE_integer('batch_size', 64, 'Batch size.')
#flags.DEFINE_float('weight_decay', 5e-4, 'Weight for regularization item')
#flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate. ')
#flags.DEFINE_float('BETA', 2., 'Beta.')
# -

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


data = load_data('C:/Users/hexie/Desktop/elstmd/data/contact.npy')

# +
data = load_data('C:/Users/hexie/Desktop/elstmd/data/contact.npy')

hist_len = 3
data_name = "contact"
nodes_num = 274
encoder = [128]
lstm = [256,256]
decoder = [274]

#historical_len= how many layers do we go back
#encoder_units = how many encoders do we expect, default set to 128
#decode_units = how many decoder units do we expect, default set to the same as num_nodes
#lstm_units, originally defulat to 256,256, we keep the same thing.
# name = dataset name
# they have splited the set of training to be from time layer 0：240，and then testing from 240:320
# we are different, we only have 7 different time layers, we set the training from 0:5 and testing 6
# basically we use their code and we change the input to adjustable to our format
# we also want to seek the output of the feature files from their code

###### IMPORTANT######
# I set everything about epoch and batch size as the same thing in whatever the origninal code is
# Acutally not true: I set the epoch to be 100 while the orignal code set it to be 1600, I did this to save running time
# I want to debug really quickly for at least one of the datasets we are using.
# 



model = e_lstm_d(num_nodes=nodes_num, historical_len=hist_len, encoder_units=[int(x) for x in encoder], 
                 lstm_units=[int(x) for x in lstm], 
                 decoder_units=[int(x) for x in decoder],
                 name=data_name)
# -
a = model._build()


a

A = a.eval(session = K.get_session())

# +
################### 6,3 model is this one, refer to this one for everything after this, this is abosultely correct####
########### if there is bug check that hist_len =3 in the model above###############

trainX = np.array([data[k: 3 +k] for k in range(1)], dtype=np.float32)
trainY = np.array(data[3:4], dtype=np.float32)
testX = np.array([data[3:6] ], dtype=np.float32)
testY = np.array(data[6:7], dtype=np.float32)
# -







history = model.train(trainX, trainY)
loss = history.history['loss']
np.save('loss.npy', np.array(loss))
aucs, err_rates, y_pres, y = model.evaluate(testX, testY)
model.save_weights('tmp/')
print(np.average(aucs), np.average(err_rates))

y_pres.shape

# +






# -

model._build()

import h5py
#tesloss = np.load('loss.npy')
#model.load_weights('./tmp/contact.h5')
f = h5py.File('./tmp/contact.h5', 'r')
f.keys()



tesweight = np.loadtxt('./tmp/contact.h5')

