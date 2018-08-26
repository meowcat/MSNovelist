# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 05:15:51 2018

@author: dvizard
"""

import numpy as np
import pickle
import smiles_config as sc



smiles_in = sc.config['smiles_full']
smiles_out = sc.config['smiles_subset']

with open(smiles_in) as d:
    smiles = d.read().split('\n')[:-1]

MAX_DATASET_SIZE = sc.config['subset_size']
smiles_train = np.random.choice(smiles, MAX_DATASET_SIZE)
np.savetxt(smiles_out, smiles_train, "%s")


