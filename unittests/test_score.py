# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:35:51 2020

@author: stravsm
"""
import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import infrastructure.score as msc
import numpy as np
import fp_management.fingerprinting as fpr



class ScoreTest(unittest.TestCase):
    
    def setUp(self):
        
        
        self.stats = fpr.load_stats(sc.config['fp_map'])
        self.fp_map = fpr.load_fp_map(self.stats)
        self.stats = self.stats[500:510,]
        
        self.predicted = np.array([1,1,1, 0.75,0.75,0.3,0.3,0.3,0,0])
        self.predicted_capped = np.array([0.9,0.9,0.9, 0.75,0.75,0.3,0.3,0.3,0.1,0.1])

        self.candidates = np.array([
            [1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0],
            [1,1,1,1,1,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,1],
            [0,1,0,1,0,1,0,1,0,1]
            ])

    def test_score_platt(self):
        score = msc.score_platt(self.predicted_capped, 
                                      self.candidates,
                                      self.stats)
        # Note: I haven't yet checked these results by hand!
        score_results = np.array([1.10716875e-04,
                            1.73643750e-05, 
                            1.13927664e-01, 
                            1.68750000e-08,
                            9.56812500e-06])
        #print(score)
        score_diff = np.sum(np.abs(score - score_results))
        self.assertTrue(score_diff < 0.0001)
        
    def test_score_unit(self):
        score = msc.score_unit(self.predicted_capped, 
                                      self.candidates,
                                      self.stats)
        # Note: This one is correct, quite trivial
        score_results = np.array([5,5,10,0,4])
        #print(score)
        score_diff = np.sum(np.abs(score - score_results))
        self.assertTrue(score_diff < 0.0001)
        
    def test_score_max_likelihood(self):
        score = msc.score_max_likelihood(self.predicted_capped, 
                                      self.candidates,
                                      self.stats)
        # Note: I haven't yet checked these results by hand!
        score_results = np.array([3.21167172e-07, 4.18618478e-05,
                                  1.18212613e-01, 1.13732799e-10,
                                  7.52378818e-07])
        #print("ML:", score)
        score_diff = np.sum(np.abs(score - score_results))
        self.assertTrue(score_diff < 0.0001)

    def test_score_mod_platt(self):
        score = msc.score_mod_platt(self.predicted_capped, 
                                      self.candidates,
                                      self.stats)
        # Note: I haven't yet checked these results by hand!
        score_results = np.array([9.79304128e-05,
                                  9.68185508e-06,
                                  6.40388209e-04,
                                  1.48058326e-06,
                                  1.65782730e-05])
        print("ML:", score)
        score_diff = np.sum(np.abs(score - score_results))
        self.assertTrue(score_diff < 0.0001)

if __name__ == '__main__':
    unittest.main()
