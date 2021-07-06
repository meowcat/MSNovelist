# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:41:27 2020

@author: stravsm
"""



import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import numpy as np
import tokens_process as tkp
import tensorflow as tf

from rdkit import Chem


class TokensProcessTest(unittest.TestCase):
    
    smiles = ['O=C1c2ccccc2-c3c1c4ccc(OC)c(OC)c4c(=O)n3-c5ccc6nc([nH]c6c5)CO',
     'O=C1OC(C)CC=CCCCCC(=O)Cc2cc(OC)cc(O)c12',
     'O=C1C(=CCCC1=C(C)C)C',
     'O=C(OC1CCC2Cc3occ(c3C(OC(=O)C(=CC)C)C2(C)C1C(=O)O)C)C=C(C)C',
     'O=C(NCC1OC(OC2C(O)C(OC3OC(CO)C(O)C(N)C3O)C(N)CC2N)C(N)CC1O)C',
     'O=C(OC)CC(c1cc2cc(OC)ccc2[nH]c1=O)C=3C(=O)C(O)=C(C(=O)C3O)C(c4cc5cc(OC)ccc5[nH]c4=O)CC(=O)OC',
     'O=C(NC1COC2C1OCC2n3nnnc3-c4cccc(OC)c4)N5CCOCC5',
     'O=[N+]([O-])c1ccc2c(nc3ccccc3c2NCCCCCCCCNc4c5ccccc5nc6cc(ccc64)[N+](=O)[O-])c1',
     'O1CCCOC1CCc2ccccc2',
     'O=C(OC)C(=O)c1c2ccccc2n3ccc4c5ccccc5[nH]c4c13',
     'O=C(O)c1cc(ccc1-c2ccc(C(=O)OC)c(Cl)c2)C',
     'O=CC(NC(=O)C(NC(=O)C(NC(=O)CC(C)C)Cc1ccc(O)cc1)CC(C)C)Cc2ccc(O)cc2',
     'O=C(CC)CCc1occ(c1)C',
     'O(c1ccc2cc(ccc2c1)C(=CC)N3CCCC3)C',
     'O=C(N)CC1NCCC1',
     'O=C(c1ccc(O)c(c1[O-])C[NH+]2CCN(CC)CC2)Cc3ccc(OC)c(OC)c3',
     'n1cnc2c(c1)ccc3[nH]c4ccc(cc4c23)C',
     'O=C(NCc1ccc(cc1)S(=O)(=O)N)CN2C=Cc3cc(OC)c(OC)cc3CC2=O',
     'S(c1ccc(cc1)C)CN2CCN(c3cccc(c3)C)CC2',
     'NCC1=CCCCC1']
    
    def test_tokens_process(self):
        tokens = tkp.tokens_from_smiles(self.smiles)
        # tokens_pad works in-place on the list
        tokens = tkp.tokens_pad(tokens)
        tokens_mat = tkp.tokens_encode(tokens)
        tokens_oh = tkp.tokens_onehot(tokens_mat)
        tokens_x, tokens_y = tkp.tokens_split_xy(tokens_oh)
        tokens_x_embed = tkp.tokens_embed_all(tokens_x)
        tokens_x_mf = tkp.tokens_encode_mf(tokens_x)
        tokens_dec = tkp.tokens_decode(tokens_y)
        tokens_crop = tkp.tokens_crop_sequence(tokens_dec)
        smiles_back = tkp.tokens_to_smiles(tokens_crop)
        
        [self.assertEqual(a, b) for a, b in list(zip(self.smiles, smiles_back))]
        
        # np.array_equal(tokens_x_mf[:,1:,9:], selfies_y[:,:-1,:])
        
    def test_tokens_tf(self):
        smiles_t = tf.convert_to_tensor(self.smiles)
        #tokens_t = tkp.tokens_from_smiles(tkp.ts_decode(smiles_t))
        tokens_e = tkp.tokens_to_tf(smiles_t)

if __name__ == '__main__':
    unittest.main()

