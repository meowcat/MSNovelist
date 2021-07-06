# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import fp_management.database as db
import fp_management.fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import infrastructure.generator as gen
import numpy as np

class GeneratorTest(unittest.TestCase):
    ds = None
    dss = None
    
    def setUp(self):
        self.fp_map = fpm.FingerprintMap(sc.config['fp_map'])
        
        self.fp_db  = db.FpDatabase.load_from_config(sc.config['db_path'])
        self.fp_train = self.fp_db.get_grp('Holdout')
        
    def test_smiles_pipeline(self):
        '''
        For now, this just tests whether smiles_pipeline() runs without errors
        on the standard training setup (get data from DB, unpack).
        
        The comparison of full vs short dataset is done to make sure
        the reshaping of unpacked fingerprints goes in the right direction,
        otherwise a transpose() would be needed.

        Returns
        -------
        None.

        '''
        dataset = gen.smiles_pipeline(
            self.fp_train, 
            batch_size = 13,
            fp_map = self.fp_map.positions)
        dataset_short = gen.smiles_pipeline(
            self.fp_train[:20], 
            batch_size = 13,
            fp_map = self.fp_map.positions)
        fingerprints_full_ = dataset["fingerprint"].take(1)
        fingerprints_full = np.array(list(fingerprints_full_.as_numpy_iterator()))
        fingerprints_short_ = dataset_short["fingerprint"].take(1)
        fingerprints_short = np.array(list(fingerprints_short_.as_numpy_iterator()))
        self.assertTrue(np.array_equal(fingerprints_full, fingerprints_short))
        # GeneratorTest.ds = dataset
        # GeneratorTest.dss = dataset_short
        
        
        
        
    def tearDown(self):
        self.fp_db.close()


if __name__ == '__main__':
    unittest.main()
