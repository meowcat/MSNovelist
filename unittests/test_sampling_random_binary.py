# -*- coding: utf-8 -*-


import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import fp_management.fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import fp_sampling.random_binary as spl
import tensorflow as tf
import numpy as np
import pandas as pd



class RandomBinarySamplingTest(unittest.TestCase):
    
    def get_generator(self):
        return tf.random.experimental.Generator.from_seed(self.seed)
    
    def setUp(self):
        self.stats = fpm.FingerprintMap(sc.config["fp_map"])
        self.fp_map = [ 967, 968, 969, 970, 971, 973, 974, 975, 976, 977,]
        #self.stats.subset_map(self.fp_map)
        self.stats.subset_map(self.fp_map, iloc=False)
        self.seed = 66
        
        self.stats_ref = np.array(
            [[0.8682724 , 0.9503411 ],
       [0.90812576, 0.921469  ],
       [0.9872968 , 0.8609618 ],
       [0.90505135, 0.9683551 ],
       [0.9648014 , 0.77676475],
       [0.6979352 , 0.9794601 ],
       [0.9819981 , 0.84515107],
       [0.763097  , 0.9775453 ],
       [0.37632978, 0.99222684],
       [0.95542455, 0.9737724 ]], dtype="float32")
        
                
        self.candidates = np.array([
            [1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1],
            [0,1,0,1,0,1,0,1,0,1]
            ], dtype="float32")
    
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
        
                
        ref_sampled = np.array(
            [[1., 1., 1., 0., 1., 1., 1., 1., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
       [0., 1., 0., 1., 1., 0., 0., 1., 0., 1.]] )
        
    def test_get_sampler(self):
        '''
        Check if generating a sampler works correctly and the stats
        are the ones we know.

        Returns
        -------
        None.

        '''
        sf = spl.SamplerFactory(sc.config)
        sampler = sf.get_sampler()
        
        
    def test_tf_random(self):
        '''
        Check that random function consistently generates our reference random array

        Returns
        -------
        None.

        '''    

        tf_random = self.get_generator().uniform(self.candidates.shape, dtype="float32")

        self.assertTrue(
            np.all(np.isclose(self.ref_random, tf_random, atol=0.0001))
            )
        
    def test_tf_random_binary(self):
        '''
        Test random binary sampling

        Returns
        -------
        None.

        '''
        rbs = spl.RandomBinarySampler(self.stats, 
                                      self.get_generator())
        # self.rbs = rbs
        # self.
        fp_sampled = rbs.sample(self.candidates)
        self.fp_sampled = fp_sampled
        
        fp_all_1 = (self.ref_random[0,:] < rbs.stats[:,0]).astype("float32")
        fp_all_0 = (self.ref_random[1,:] > rbs.stats[:,1]).astype("float32")
        
        self.assertTrue(np.array_equiv(fp_sampled.numpy()[0,:], fp_all_1))
        self.assertTrue(np.array_equiv(fp_sampled.numpy()[1,:], fp_all_0))
        
        ref_sampled = np.array(
           [[1., 1., 1., 0., 1., 1., 1., 1., 0., 1.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
       [0., 1., 0., 1., 1., 0., 0., 1., 0., 1.]] )
        
        self.assertTrue(np.array_equal(fp_sampled, ref_sampled))
        

        

if __name__ == '__main__':
    unittest.main()
