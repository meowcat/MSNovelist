# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 09:17:44 2018

@author: stravsmi
"""


import smiles_process as sp
import numpy as np
#import h5py
import shutil
import pickle
from tables import *
import tables
import smiles_config as sc

"""
file_in = "C:/Cloud/SWITCHdrive/CVAE-molecule/CVAE-master-data/smiles.txt"
file_out = "C:/Cloud/SWITCHdrive/CVAE-molecule/CVAE-master-data/smiles-train.h5"
file_in = "/storage/smiles/smiles.txt"
file_temp = "/smiles_train.h5"
file_out = "/storage/smiles/smiles_train.h5"
#f = "/storage/smiles/smiles.txt"
d = open(file_in)
smiles = d.read().split('\n')[:-1]
MAX_DATASET_SIZE = 40000
smiles_train = np.random.choice(smiles, MAX_DATASET_SIZE)
"""



file_in = sc.config['smiles_subset']
file_out = sc.config['smiles_encoded']


with open(file_in) as d:
    smiles_train = d.read().split('\n')[:-1]


# find dimensions of single transformed smiles to provision the file
shape_blueprint_X, shape_blueprint_y, l = sp.SmilesPreprocessingPipeline.transform(smiles_train[0])
"""
h5f = h5py.File(file_temp, 'w')
smiles_ds = h5f.create_dataset('smiles_Xy', 
                   shape=(0,shape_blueprint.shape[1], shape_blueprint.shape[2]),
                   maxshape=(None, shape_blueprint.shape[1], shape_blueprint.shape[2]))
"""

f = tables.open_file(file_out, 'w')
smiles_X_eary = f.create_earray(f.root, "smiles_X", 
                               tables.Atom.from_dtype(shape_blueprint_X.dtype),
                               (0, shape_blueprint_X.shape[1], shape_blueprint_X.shape[2]))

smiles_y_eary = f.create_earray(f.root, "smiles_y", 
                               tables.Atom.from_dtype(shape_blueprint_y.dtype),
                               (0, shape_blueprint_y.shape[1]))
l = np.array([])
BLOCK_SIZE = 10

# TEST_SPLIT = 0.2
# unused for now

#for i in range(BLOCK_COUNT):
i=0
smiles_len = len(smiles_train)
for block_start in range(0, smiles_len, BLOCK_SIZE):
    smiles_X_block, smiles_y_block, l_block = sp.SmilesPreprocessingPipeline.transform(
            smiles_train[block_start:block_start+BLOCK_SIZE])
    #smiles_ds.resize(smiles_ds.shape[0]+smiles_Xy_block.shape[0], axis=0)
    #smiles_ds[-smiles_Xy_block.shape[0]:] = smiles_Xy_block
    smiles_X_eary.append(smiles_X_block)
    smiles_y_eary.append(smiles_y_block)
    l = np.r_[l, l_block]
    i+=1
    print("Block ", i, "processed")
    block_start += BLOCK_SIZE

f.create_array(f.root, "smiles_counts", l)

f.close()

"""
f = tables.open_file(file_out, 'r')
smiles_Xy_loaded = f.root.smiles_X[:]
"""

#h5f.close()



