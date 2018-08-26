# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 11:27:38 2018

@author: stravsmi
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools

from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import MACCSkeys

import numpy as np

import pickle

import smiles_config as sc

smiles_in = sc.config['smiles_subset']
fp_out = sc.config['smiles_fingerprints']


with open(smiles_in) as d:
    smiles_train = d.read().split('\n')[:-1]


mols_train = [Chem.MolFromSmiles(i) for i in smiles_train]
fp_train = [Chem.RDKFingerprint(i) for i in mols_train]
maccs_train = [MACCSkeys.GenMACCSKeys(i) for i in mols_train]
avalon_train = [pyAvalonTools.GetAvalonFP(i) for i in mols_train]
    
fps = [np.r_[ fp_train[i], maccs_train[i], avalon_train[i]] for i in range(len(mols_train))]
    
#fpbits = [i for i in Chem.RDKFingerprint(ms[1])]

avalon_train_bin = np.array(avalon_train)
fp_train_bin = np.array(fp_train)
maccs_train_bin = np.array(maccs_train)

fp3_train = np.c_[fp_train_bin, maccs_train_bin, avalon_train_bin]
with open(fp_out, 'wb') as output:
    pickle.dump(fp3_train, output, pickle.HIGHEST_PROTOCOL)
