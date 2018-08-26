# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:37:22 2018

@author: stravsmi
"""

import smiles_process as sp
import numpy as np


f = "C:/Cloud/SWITCHdrive/CVAE-molecule/CVAE-master-data/smiles.txt"

d = open(f)
smiles_raw = d.read()
smilesChars = set(smiles_raw)

# We want to transform all two-letter atoms:
# Cl to L
smiles_raw.count('Cl')
smiles_raw.count('C')
smiles_raw.count('l')
# Br to R (note that we have no boron things in the dataset)
smiles_raw.count('Br')
smiles_raw.count('B')
smiles_raw.count('r')
# Fe - there is none


smilesChars_new = set(sp.smiles_mapElements(smiles_raw))


smiles = d.read().split('\n')[:-1]

EXPLORE_SIZE = 5

smiles_exp = np.random.choice(smiles, EXPLORE_SIZE)
smiles_cat = sp.SmilesPreprocessingPipeline.transform(smiles_exp)

X = smiles_cat[:,:sp.SEQUENCE_LEN,:]
y = smiles_cat[:,sp.SEQUENCE_LEN,:]



#smiles_long = smiles_exp.reshape(1)
#smiles_set = set(smiles_exp)

smiles_tf = []
for s in smiles_exp:
    smiles_tf.append(Chem.MolToSmiles(Chem.MolFromSmiles(s),
                     allBondsExplicit=False, isomericSmiles=False))
(smiles_exp, smiles_tf)

smilesLenDist = [len(s) for s in smiles]

import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1.hist(smilesLenDist)