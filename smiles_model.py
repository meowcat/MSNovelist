# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:58:01 2018

@author: stravsmi
"""


import smiles_process as sp
import numpy as np

from tensorflow.keras.layers import Input, LSTM, concatenate, add, CuDNNLSTM, RepeatVector
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

import pickle
import tables
import time

import smiles_config as sc

start_time = str(int(time.time()))

in_fingerprints = sc.config['smiles_fingerprints']
in_smiles = sc.config['smiles_encoded']



# load smiles training data
with tables.open_file(in_smiles, 'r') as h5f:
    smiles_X = h5f.root.smiles_X[:].astype("float32")
    smiles_y = h5f.root.smiles_y[:].astype("float32")
    smiles_counts = h5f.root.smiles_counts[:].astype(int)

# load fp training data
with open(in_fingerprints, "rb") as pkl:
    fp_X = pickle.load(pkl)
    
X_seq_len = np.argmax(smiles_X[:,:,sp.smiles_ctoi[sp.PAD_CHAR]], axis=1)



fp_X_long = np.repeat(fp_X, smiles_counts.astype(int), axis=0).astype("float32")


#fp_X = np.array([i for i in range(107)])
#fp_X = np.c_[fp_X, fp_X, fp_X]

if(sc.config['rnn_cell'] == "LSTM"):
    cell = LSTM
if(sc.config['rnn_cell'] == "CuDNNLSTM"):    
    cell = CuDNNLSTM
rnn_hidden_size = sc.config['rnn_hidden_size']
#layers = sc.config['layers']




input_fp = Input(shape=(fp_X_long.shape[1],), name="fp_input")

fp_layer = Dense(sc.config['fp_embedding_size'])(input_fp)
#fp_layer = Dropout(sc.config['fp_dropout'])(fp_layer)
fp_layer = RepeatVector(smiles_X.shape[1])(fp_layer)

input_smiles = Input(shape=(smiles_X.shape[1], smiles_X.shape[2]))
input_merge = concatenate([fp_layer, input_smiles])

rnn_layer = cell(sc.config['rnn_hidden_size'], return_sequences=True,  activation="relu")(input_merge)
rnn_layer = Dropout(sc.config['rnn_dropout'])(rnn_layer)
rnn_layer = cell(sc.config['rnn_hidden_size'], return_sequences=True, activation="relu")(rnn_layer)
rnn_layer = Dropout(sc.config['rnn_dropout'])(rnn_layer)
out_layer = Dense(smiles_y.shape[1], activation="softmax")(rnn_layer)

model = Model(inputs=[input_fp, input_smiles], outputs=[out_layer])
model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=['accuracy'])

filepath=sc.config["weights_folder"] + "model" + start_time + "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit([fp_X_long, smiles_X], smiles_y, 
          epochs=sc.config['epochs'], 
          batch_size=sc.config['batch_size'],
          callbacks = callbacks_list)



#input_smiles = Input(shape=(smiles_X_flat.shape[1],), dtype="bool", name="smiles_input")
#layer_ =add([input_fp, input_smiles])
#layer_ = cell(cell_size, return_sequences=True)(layer_)
#layer_ = cell(cell_size)(layer_)
#layer_ = cell(cell_size)(layer_)
