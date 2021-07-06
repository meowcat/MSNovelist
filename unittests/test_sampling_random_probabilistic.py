# -*- coding: utf-8 -*-


import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import fp_management.fingerprinting as fpr
import fp_sampling.random_probabilistic as spl
import tensorflow as tf
import numpy as np
import pandas as pd



class RandomProbabilisticSamplingTest(unittest.TestCase):
    
    def get_generator(self):
        return tf.random.experimental.Generator.from_seed(self.seed)
    
    def setUp(self):
        
        
        self.fp_true = np.array([
            [1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1]], dtype="float32")
            
        
        fp_predicted = 0.1 + self.fp_true * 0.8
        fp_predicted = fp_predicted + 0.0001*(np.arange(10) + 1).reshape((1,10))
        fp_predicted = fp_predicted + 0.01*(np.arange(4) + 1).reshape((4,1))
        self.fp_predicted = np.round(fp_predicted, 4)        
    
        self.ref_random = np.array(
        [[0.8007604 , 0.34483027, 0.30735934, 0.9700769 , 0.43336582,
        0.51829636, 0.8555572 , 0.21962428, 0.76091194, 0.04762888],
       [0.5131633 , 0.97339594, 0.33341944, 0.03152311, 0.28874612,
        0.58736026, 0.75669694, 0.7060174 , 0.5588795 , 0.40088904],
       [0.8377521 , 0.7651175 , 0.5402924 , 0.52445555, 0.8965409 ,
        0.4253844 , 0.6824161 , 0.17499602, 0.4163201 , 0.8321508 ],
       [0.313666  , 0.15126252, 0.23609889, 0.56451845, 0.44649565,
        0.41026497, 0.60833323, 0.50973356, 0.31043375, 0.07893777],
       [0.57783425, 0.16236854, 0.63097   , 0.7254863 , 0.8393048 ,
        0.89883757, 0.64980876, 0.0960505 , 0.06560767, 0.6030959 ]])
        
        
        
    def test_get_sampler(self):
        '''
        Check if generating a sampler works correctly - 
        but this is too long right now, it takes a minute.
        
        Returns
        -------
        None.

        '''
        sf = spl.SamplerFactory(sc.config)
        sampler = sf.get_sampler()
        pass
    
    def test_random_probabilistic(self):
        sampler = spl.RandomProbabilisticSampler(self.fp_true, self.fp_predicted, 0)
        fp_sampled = sampler.sample(self.fp_true)
        # this works, I just have to write a nice confirmation
        

if __name__ == '__main__':
    unittest.main()
