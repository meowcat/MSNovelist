# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 11:41:27 2020

@author: stravsm
"""
import pathlib
import os
import tempfile
import sys
import gzip

sys.path.append(os.environ['MSNOVELIST_BASE'])

import unittest

import smiles_config as sc
import numpy as np
import tokens_process as tkp
import tensorflow as tf
import shutil

from rdkit import Chem

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm





class DatabaseTest(unittest.TestCase):
    
    resources_path = pathlib.Path(
        pathlib.Path(__file__).parent,
        "resources")
        
    databases_to_test = {
        'h5_plain': {
            'path':  pathlib.Path(
                    resources_path,
                    "db_h5_dummy.h5")
            },
        'h5_with_sample': {
            'path':  
                pathlib.Path(
                    resources_path,
                    "db_h5_dummy.h5"),
            'fingerprints_degraded':  
                pathlib.Path(
                    resources_path,
                    "db_h5_sample_dummy.h5")
            },
        'sqlite': {
            'path':  pathlib.Path(
                    resources_path,
                    "db_sqlite_dummy.db")
            },
        'csv': {
            'path': '',
            'fp_map':  pathlib.Path(
                os.environ['MSNOVELIST_BASE'],
                "data", "fingerprint-map", "csi_fingerid.csv"),
            'construct_from': 'smiles'

            }        
        }
    
    temp_path = ''
    
    def setUp(self):
        
        # Unzip the csv database to a temp folder
        unzip_target = tempfile.mkdtemp()
        db_target = pathlib.Path(
                    unzip_target,
                    "db_csv_dummy.csv")
        with gzip.open(
                pathlib.Path(
                    self.resources_path,
                    'db_csv_dummy.csv.gz'), 'r') as f:
            file_content = f.read()
        
        with open(db_target
                , "wb") as f:
            f.write(file_content)
        self.databases_to_test['csv']['path'] = db_target
        self.temp_path = unzip_target
        
        # Initialize the fingerprinter
        fp_map = fpm.FingerprintMap(sc.config["fp_map"])
        fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = False)
        
    def test_sqlite(self):
        db_test = db.FpDatabase.load_from_config(self.databases_to_test['sqlite'])
        db_grp_test = db_test.get_grp("fold1")
        self.assertEqual(len(db_grp_test), 12)
    
    def test_csv(self):
        db_test = db.FpDatabase.load_from_config(self.databases_to_test['csv'])
        db_grp_test = db_test.get_grp("fold0")
        self.assertEqual(len(db_grp_test), 4)
    
    def test_h5_plain(self):
        db_test = db.FpDatabase.load_from_config(self.databases_to_test['h5_plain'])
        db_grp_test = db_test.get_grp("fold0")
        self.assertEqual(len(db_grp_test), 2)
        db_grp_test = db_test.get_grp("fold1")
        self.assertEqual(len(db_grp_test), 8)
        # Compare a fp_degraded to a fp and find that they are identical
        entry_1 = db_grp_test[0]
        self.assertTrue(np.array_equiv(
            entry_1["fingerprint"], entry_1["fingerprint_degraded"])
            )
    
    def test_h5_with_sample(self):
        db_test = db.FpDatabase.load_from_config(self.databases_to_test['h5_with_sample'])
        db_grp_test = db_test.get_grp("fold0")
        self.assertEqual(len(db_grp_test), 2)
        db_grp_test = db_test.get_grp("fold1")
        self.assertEqual(len(db_grp_test), 8)
        # Compare a fp_degraded to a fp and find that they are different (because we loaded a sample)
        entry_1 = db_grp_test[0]
        self.assertFalse(np.array_equiv(
            entry_1["fingerprint"], entry_1["fingerprint_degraded"])
            )
    
    
    def tearDown(self):
        # pass
        shutil.rmtree(
            self.temp_path
            )
        
        
                

if __name__ == '__main__':
    unittest.main()

