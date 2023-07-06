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
import numpy as np
import fp_management.fingerprinting as fpr

import fp_management.database as db
import fp_management.fingerprint_map as fpm

from matplotlib import pyplot as plt

import tempfile
import pandas as pd
import yaml

test_cases = """
chelidonine:
  smiles: "CN1CC2=C(C=CC3=C2OCO3)C4C1C5=CC6=C(C=C5CC4O)OCO6"
  name: chelidonine
  bits_set: [ 18, 26, 44, 66, 77, 333, 418, 447, 454, 458, 465, 481, 482, 488, 489, 499, 506, 509, 514, 677, 684, 710, 712, 714, 720, 727, 785, 813, 814, 868, 874, 893, 1072, 1135, 1166, 1183, 1184, 1190, 1210, 1224, 1226, 1228, 1230, 1231, 1240, 1283, 1284, 1304, 1305, 1325, 1326, 1328, 1410, 1708, 3054, 4389, 4391, 4789, 4823, 4968, 4969, 4999, 5088, 5090, 5111, 5118, 5141, 5149, 5695, 5701, 6228, 6432, 6553, 6617, 6855, 7047, 7307, 7444, 7577, 7621, 7726, 7846, 7922, 8088, 8093, 8202, 8219, 8228, 8230, 8232, 8240, 8248, 8254, 8261, 8276, 8297, 8302, 8310, 8311, 8315, 8321, 8327, 8332, 8340, 8341, 8606, 8857, 8880, 8984, 9031, 9058, 9184, 9202, 9239, 9252, 9370, 9394, 9413, 9441, 9465, 9657, 9762, 9783, 9885, 10048, 10058, 10070, 10182, 10183, 10186, 10266, 10286, 10343, 10353, 10413, 10620, 10666, 10694, 10725, 10726, 10776, 10817, 10833, 10839, 10908, 10958, 11080, 11173, 11186, 11197, 11204, 11253, 11309, 11390, 11404, 11419, 11499, 11517, 11653, 11689, 11701, 11765, 11795, 11797, 11834, 11970, 11996, 12080, 12157, 12199, 12209, 12216, 12270, 12277, 12446, 12524, 12545, 12620, 12841, 12875, 12915, 13010, 13331]
fluorocompound:
  smiles: "N[C@@](F)(C)C(=O)O"
  smiles_flat: "O=C(O)C(F)(N)C"
  name: fluorothing
  bits_set: [ 26, 63, 77, 415, 445, 456, 473, 492, 495, 813, 814, 868, 1064, 2564, 3668, 4863, 5703, 7302, 8433, 8441, 8454, 9017, 9098, 9214, 9771, 10024, 10048, 10243, 10250, 10623, 10766, 11609, 12813 ]  

"""

class FingerprintingTest(unittest.TestCase):
    
    def setUp(self):
        self.cache = tempfile.mkstemp(suffix = '.db')
        os.close(self.cache[0])
        self.fp_map = fpm.FingerprintMap(sc.config['fp_map'])
        fpr.Fingerprinter.init_instance(
            sc.config['fingerprinter_path'],
            self.fp_map,
            sc.config['fingerprinter_threads'],
            capture = True,
            cache = self.cache[1])
        self.fingerprinter = fpr.Fingerprinter.get_instance()
        self.data = yaml.safe_load(test_cases)

        
    def tearDown(self):
        self.fingerprinter.cache.close()
        os.remove(self.cache[1])

    def test_flatten_smiles(self):
        smiles = self.data["fluorocompound"]["smiles"]
        smiles_processed = self.fingerprinter.process([smiles], 
                                                      calc_fingerprint = False)
        self.assertEqual(len(smiles_processed), 1)
        self.assertEqual(smiles_processed[0]["data_id"], 0)
        self.assertEqual(smiles_processed[0]["smiles_canonical"],
                         self.data["fluorocompound"]["smiles_flat"])
        
    def test_calc_fingerprint(self):
        for _, data in self.data.items():
            result_array = self.fingerprinter.process(
                [data["smiles"]], 
                return_b64 = False,
                return_numpy = True
                )
            fp_array = result_array[0]["fingerprint"]
            bits_set = np.array(data['bits_set'])

            self.assertTrue( (fp_array[0,bits_set] == 1).all() )
            fp_array[0,bits_set] = 0
            self.assertTrue( (fp_array == 0).all() )


        
    def test_multiple_fingerprints(self):
        smiles = ["CCCCC", "CCCCC(O)CC", "CCCCC(CCC(O)C)C"]  * 10
        emptyPos = [5,10,15,20]
        #emptyPos = []
        for pos in emptyPos:
            smiles[pos] = "not a smiles" + str(pos)
        smiles_processed = self.fingerprinter.process(smiles, calc_fingerprint = True, return_b64 = True)
        for ids in [0,1,5,9,12,15,29]:
            self.assertEqual(smiles_processed[ids]["data_id"], ids)
        for ids in emptyPos:
            self.assertEqual(smiles_processed[ids]["fingerprint"], None)
            self.assertEqual(smiles_processed[ids]["smiles_generic"], "")
        for ids in [8,11,14]:
            self.assertEqual(smiles_processed[2]["fingerprint"],
                             smiles_processed[ids]["fingerprint"])
            self.assertEqual(smiles_processed[2]["smiles_generic"],
                             smiles_processed[ids]["smiles_generic"])
            
    def test_consume_fingerprinters(self):
        '''
        Check that failing SMILES don't break the fingerprinter. We should have
        2*sc.config['fingerprinter_threads'] fingerprinters at the beginning,
        we try to exhaust them by running fingerprinting with gugus smiles
        many times.

        Returns
        -------
        None.

        '''
        smiles = ["haha", "N[C@@](F)(C)C(=O)O", "gaga", "gaga", "gugus"]
        for _ in range(sc.config['fingerprinter_threads']):
            smiles_processed = self.fingerprinter.process(smiles)
            
            #self.assertEqual(smiles_processed[1]["fingerprint"][:20],
            #                  'BAAADACggoAAYAAAAAAw')

            # Affaire à suivre, used to be 'BAAADACgAgABAcAAAAAA' - is this
            # all from parsing SMILES instead of InChI?
            # No, the difference is made by the fact that the first
            # 55 bits are OpenBabel. In the old setup, the second
            # fingerprint was aligned from 64 upwards. Now it's continuing
            # straight from pos 55.
            self.assertEqual(smiles_processed[0]["fingerprint"], None)
            self.assertEqual(smiles_processed[0]["smiles_generic"], "")
            self.assertNotEqual(smiles_processed[1]["fingerprint"], None)
            self.assertNotEqual(smiles_processed[1]["smiles_generic"], "")
            self.assertEqual(smiles_processed[3]["fingerprint"], None)
            self.assertEqual(smiles_processed[3]["smiles_generic"], "")

    @unittest.skip("everything changed, we don't use jpype anymore")
    def test_fingerprint_alignment(self):
        '''
        This generates the fingerprint for chelidonine,
        and checks that bits as obtained by get_fp are the same value
        as if obtained directly from Java.

        Returns
        -------
        None.

        '''            
        smi_CHE = self.smi_CHE
        # These bits must be TRUE for chelidonine.
        # Get the indices of set bits for chelidonine directly out of the Java class
        fp_true = self.reference_indices(smi_CHE)
        
        #FP_TRUE = [45,349,1192,1092,1018,880,1168,6594,6416,7059]
        # Some of the ECFP chelidonine bits have changed...
        
        fp_false = [i+1 for i in  fp_true] + [i-1 for i in  fp_true]
        fp_false = [i for i in fp_false if i not in fp_true]
        # compute the fingerprint with SIRIUS/Java
        f = self.fingerprinter.process([smi_CHE])
        fp = fpr.get_fp(f[0]["fingerprint"])
        # Check the correct length
        self.assertEqual(fp.shape, (1,self.fingerprinter.fp_len))
        # check if all bits are correct
        self.assertTrue(all(fp[0,fp_true]))
        self.assertFalse(any(fp[0,fp_false]))
        
    
    def reference_indices(self, smiles = None):
        if smiles is None:
            smiles = self.smi_CHE
        test_fingerprint_chelidonine = \
            self.fingerprinter.fpu.getTestFingerprint(smiles)
        test_fingerprint = test_fingerprint_chelidonine[0]
        indices = [i for i in test_fingerprint.toIndizesArray()]
        return indices
    


    def test_b64_equivalent(self):
        ref = ['C=CC(C)(C)c1cc2cc3ccoc3cc2oc1=O',
                  'CC1(C)CC(c2c(cc(cc2O1)OCC(Nc3ccc(cc3)S(N)(=O)=O)=O)O)=O',
                  'C=C(C)C(C)C1C(C(C)(C2C3CCC4C5(C)CCC(C(C)(C)C5CCC4(C)C63CC2(OC6)O1)OC7C(C(C(CO7)O)OC8C(C(C(C(CO)O8)O)OC9C(C(C(CO9)O)O)O)O)OC%10C(C(C(CO)O%10)O)O)O)OC(C)=O',
                  'Cc1cc(c2c(C)c(C)c(=O)oc2c1)OCC(N3CCC(CC3)C(=N)O)=O',
                  'CCCn1c2c(c(n(CCC)c1=O)=O)[nH]c(-c3ccc(cc3)OCC(Nc4ccc(cc4)C#N)=O)n2',
                  'CC1C=CC2CCCCC2C1CCC3CC(CC(=O)O3)O',
                  'Cn1c2c(c(n(C)c1=O)=O)[nH]c(-c3ccc(cc3)S(O)(=O)=O)n2',
                  'CC1C(C(C2C3(C)C(CC4C2(C)C1(C(C(=O)O4)O)O)C(C)(CC(C3O)O)O)O)O',
                  'c1ccc(cc1)CN=C(CCC(N2CC3CC(C2)c4cccc(n4C3)=O)=O)O',
                  'c1c(cc(c(c1O)O)O)C(=O)OCC2C(C(C(C(O2)OC(c3cc(c(c(c3)O)O)O)=O)O)OC(c4cc(c(c(c4)O)O)O)=O)O',
                  'CC(C)=CCc1cc2C(C(COc2c(CC=C(C)C)c1O)c3ccc(cc3O)O)=O',
                  'c1cc2c(cc1O)c(CC(C(=O)O)N)c[nH]2',
                  'COc1ccc2c(c1)OCC3c4ccc(cc4OC23)OC',
                  'COc1cc(ccc1O)C=CC(=NCCc2ccc(cc2)O)O',
                  'CN1CCc2cc3c(cc2C1C4c5ccc6c(c5C(=O)O4)OCO6)OCO3',
                  'c1ccc(cc1)C=CC(=NCCc2cnc[nH]2)O']
        processed_b64 = self.fingerprinter.process(ref)
        processed_bytes = self.fingerprinter.process(ref, return_b64 = False)
        fp_from_b64 = np.array([fpr.get_fp(x["fingerprint"], b64decode = True) 
                        for x in processed_b64])
        fp_from_bytes = np.array([fpr.get_fp(x["fingerprint"], b64decode = False) 
                        for x in processed_bytes])
        self.assertTrue(np.array_equal(fp_from_b64, fp_from_bytes))

    def test_cache(self):
        ref = ['C=CC(C)(C)c1cc2cc3ccoc3cc2oc1=O',
                  'failtest1',
                  np.nan,
                  'C=C(C)C(C)C1C(C(C)(C2C3CCC4C5(C)CCC(C(C)(C)C5CCC4(C)C63CC2(OC6)O1)OC7C(C(C(CO7)O)OC8C(C(C(C(CO)O8)O)OC9C(C(C(CO9)O)O)O)O)OC%10C(C(C(CO)O%10)O)O)O)OC(C)=O',
                  'Cc1cc(c2c(C)c(C)c(=O)oc2c1)OCC(N3CCC(CC3)C(=N)O)=O',
                  'CCCn1c2c(c(n(CCC)c1=O)=O)[nH]c(-c3ccc(cc3)OCC(Nc4ccc(cc4)C#N)=O)n2',
                  'CC1C=CC2CCCCC2C1CCC3CC(CC(=O)O3)O',
                  'Cn1c2c(c(n(C)c1=O)=O)[nH]c(-c3ccc(cc3)S(O)(=O)=O)n2',
                  'CC1C(C(C2C3(C)C(CC4C2(C)C1(C(C(=O)O4)O)O)C(C)(CC(C3O)O)O)O)O',
                  'c1ccc(cc1)CN=C(CCC(N2CC3CC(C2)c4cccc(n4C3)=O)=O)O',
                  'c1c(cc(c(c1O)O)O)C(=O)OCC2C(C(C(C(O2)OC(c3cc(c(c(c3)O)O)O)=O)O)OC(c4cc(c(c(c4)O)O)O)=O)O',
                  'CC(C)=CCc1cc2C(C(COc2c(CC=C(C)C)c1O)c3ccc(cc3O)O)=O',
                  'c1cc2c(cc1O)c(CC(C(=O)O)N)c[nH]2',
                  'COc1ccc2c(c1)OCC3c4ccc(cc4OC23)OC',
                  'COc1cc(ccc1O)C=CC(=NCCc2ccc(cc2)O)O',
                  'CN1CCc2cc3c(cc2C1C4c5ccc6c(c5C(=O)O4)OCO6)OCO3',
                  'c1ccc(cc1)C=CC(=NCCc2cnc[nH]2)O',
                  'failtest2']
        df = pd.DataFrame({"smiles": ref})
        df = db.process_df(df, self.fingerprinter, construct_from="smiles")
        df_fill_cache = self.fingerprinter.process_df(df.iloc[:5].copy(), verbose_cache_column = "cached")
        self.assertTrue(sum(df_fill_cache["cached"]) == 0)
        df_all = self.fingerprinter.process_df(df, verbose_cache_column = "cached")
        self.assertTrue(sum(df_all["cached"]) == sum(df_fill_cache["fingerprint"].notna()))
        self.assertFalse("fingerprint" in df.columns)
        self.assertTrue("fingerprint" in df_all.columns)
        # Test another output column
        df_all = self.fingerprinter.process_df(df, verbose_cache_column = "alternative_cached",
                                               out_column = "alternative_fingerprint")
        # test another input column
        df_alt = pd.DataFrame({"smiles": ref})
        df_alt = db.process_df(df_alt, self.fingerprinter, construct_from="smiles")
        df_alt.rename(columns={"smiles": "alternative_smiles"}, inplace=True)
        self.assertRaises(
            KeyError,
            lambda: self.fingerprinter.process_df(df_alt))
        df_alt_fill = self.fingerprinter.process_df(df_alt, in_column = "alternative_smiles")
        self.assertFalse("fingerprint" in df_alt.columns)
        self.assertTrue("fingerprint" in df_alt_fill.columns)
        # Test inplace handling with regular and alternative columns
        self.fingerprinter.process_df(df_alt, in_column = "alternative_smiles", inplace=True)
        self.assertTrue("fingerprint" in df_alt.columns)
        self.assertFalse("cached" in df.columns)
        self.fingerprinter.process_df(df, out_column = "alt_fp", 
                                      verbose_cache_column = "cached",
                                      inplace=True)
        self.assertFalse("fingerprint" in df.columns)
        self.assertTrue("alt_fp" in df.columns)
        self.assertTrue("cached" in df.columns)
        # Remove the cache and test without cache
        cache_ = self.fingerprinter.cache
        self.fingerprinter.cache = None
        df = pd.DataFrame({"smiles": ref})
        df = db.process_df(df, self.fingerprinter, construct_from="smiles")
        self.fingerprinter.process_df(df, verbose_cache_column = "cached", inplace=True)
        self.assertTrue(sum(df["cached"]) == 0)
        self.fingerprinter.cache = cache_
        
        
    def not_test_consistency(self):
        data_eval_ = os.path.join(sc.config['base_folder'],
                      "evaluation_v44/dataset2/predictions.csv")
        fp_map = fpm.FingerprintMap(sc.config['fp_map'])
        db_eval = db.FpDatabase.load_from_config({
            'path': data_eval_, 
            'fp_map': sc.config['fp_map'],
            'nrows': 1000,
            'construct_from': 'inchi'}
            )
        
        processed_canonical = self.fingerprinter.process(
            db_eval.data_information.smiles_canonical)
        
        # Then from the RDKit-generated "smiles input" (which however should
        # be canonicalized by CDK before fingerprinting)
        processed_smiles_in = self.fingerprinter.process(
            db_eval.data_information.smiles_in)
        fingerprints_smiles_in = np.concatenate(
            [fpr.get_fp(data["fingerprint"]) for data in processed_smiles_in])
        fingerprints_smiles_in = fingerprints_smiles_in[:,fp_map.positions]
        
        # And get the reference fingerprints from the loaded database        
        fingerprints_ref = db_eval.data_fp_true[:,fp_map.positions]
        
        
        # Get the SMILES from Pubchem and run that through the fingerprinter,
        # except for failures
        df_smiles_pubchem = db.get_smiles_pubchem(
            db_eval.data_information, db.db_pubchem)
        proc_smiles_pubchem = self.fingerprinter.process(df_smiles_pubchem.smiles_in)
        fingerprints_smiles_pubchem = np.concatenate(
            [fpr.get_fp(data["fingerprint"]) for data in proc_smiles_pubchem])
        fingerprints_ref_ok = fingerprints_ref[df_smiles_pubchem["smiles_in"] != "",:]
        fingerprints_smiles_pubchem = fingerprints_smiles_pubchem[:,fp_map.positions]
        fingerprints_smiles_pubchem = fingerprints_smiles_pubchem[df_smiles_pubchem["smiles_in"] != "",:]
        diff_sum = np.sum(fingerprints_smiles_pubchem != fingerprints_ref_ok, axis=0)

        return
    
        
        # Process the database, which makes SMILES from the InChIs
        # but not fingerprints
        inchikey_ref = db_eval.data_information["inchikey"].to_numpy()
        db_eval.data_information["inchikey"] = ""
        db_eval.process_smiles()
        inchikey_proc = db_eval.data_information["inchikey"].to_numpy()
    
        # Check that all inchikeys stayed the same (which would be quite awesome,
        # since the original inchi was given by Kais pipeline / cdk and mine is 
        # calculated with RDKit
        self.assertEqual(np.sum(inchikey_ref != inchikey_proc), 0)
        
        # Now calculate the fingerprints and check if they come out the same
        # as in the database CSV read-in (the ZeroOneString)
        
        # First from the resulting (sic) canonical SMILES
        processed_canonical = self.fingerprinter.process(
            db_eval.data_information.smiles_canonical)
        fingerprints_canonical = np.concatenate(
            [fpr.get_fp(data["fingerprint"]) for data in processed_canonical])
        fingerprints_canonical = fingerprints_canonical[:,fp_map.positions]
        
        # Then from the RDKit-generated "smiles input" (which however should
        # be canonicalized by CDK before fingerprinting)
        processed_smiles_in = self.fingerprinter.process(
            db_eval.data_information.smiles_in)
        fingerprints_smiles_in = np.concatenate(
            [fpr.get_fp(data["fingerprint"]) for data in processed_smiles_in])
        fingerprints_smiles_in = fingerprints_smiles_in[:,fp_map.positions]
        
        # And get the reference fingerprints from the loaded database        
        fingerprints_ref = db_eval.data_fp_true[:,fp_map.positions]
        
        reproc = db_eval.data_information
        reproc["smiles"] = reproc["smiles_canonical"]
        reproc = db.process_df(reproc, self.fingerprinter, construct_from="smiles", write="inchikey")

        # Get the SMILES from Pubchem and run that through the fingerprinter,
        # except for failures
        df_smiles_pubchem = db.get_smiles_pubchem(
            db_eval.data_information, db.db_pubchem)
        proc_smiles_pubchem = self.fingerprinter.process(df_smiles_pubchem.smiles_in)
        fingerprints_smiles_pubchem = np.concatenate(
            [fpr.get_fp(data["fingerprint"]) for data in proc_smiles_pubchem])
        fingerprints_ref_ok = fingerprints_ref[df_smiles_pubchem["smiles_in"] != "",:]
        fingerprints_smiles_pubchem = fingerprints_smiles_pubchem[:,fp_map.positions]
        fingerprints_smiles_pubchem = fingerprints_smiles_pubchem[df_smiles_pubchem["smiles_in"] != "",:]
        diff_sum = np.sum(fingerprints_smiles_pubchem != fingerprints_ref_ok, axis=0)

        diff_sum_2 = np.sum(fingerprints_smiles_pubchem != fingerprints_ref_ok, axis=1)
        
        fingerprints_tp = (fingerprints_ref_ok == 1) & (fingerprints_smiles_pubchem == 1)
        fingerprints_fp = (fingerprints_ref_ok == 0) & (fingerprints_smiles_pubchem == 1)
        fingerprints_fn = (fingerprints_ref_ok == 1) & (fingerprints_smiles_pubchem == 0)
        fingerprints_tn = (fingerprints_ref_ok == 0) & (fingerprints_smiles_pubchem == 0)
        
        fingerprints_tp_sum = np.sum(fingerprints_tp, axis=0)
        fingerprints_fp_sum = np.sum(fingerprints_fp, axis=0)
        fingerprints_fn_sum = np.sum(fingerprints_fn, axis=0)
        fingerprints_tn_sum = np.sum(fingerprints_tn, axis=0)
        
        fingerprints_tpr = (fingerprints_tp_sum + 0.5) / (fingerprints_tp_sum + fingerprints_fn_sum + 0.5)
        fingerprints_tnr = (fingerprints_tn_sum + 0.5) / (fingerprints_tn_sum + fingerprints_fp_sum + 0.5)

        
        # inchikey_reproc = reproc["inchikey"].to_numpy()
        # self.assertEqual(np.sum(inchikey_ref != inchikey_proc), 0)
        
        # self.assertEqual(
        #     np.sum(fingerprints_smiles_in != fingerprints_canonical), 0)
        # self.assertEqual(
        #     np.sum(fingerprints_canonical != fingerprints_ref), 0)
        # diff_sum = np.sum(fingerprints_canonical != fingerprints_ref, axis=0)
        plt.bar(range(len(diff_sum)), diff_sum)
        
        plt.plot(fingerprints_tpr)        
        plt.hist(fingerprints_tpr)
        plt.plot(fingerprints_tnr)        
        plt.scatter(fingerprints_tpr, fingerprints_tnr)        
        # For debugging:
        self.db_eval = db_eval


# res = self.fingerprinter.process(["CCCCCCCCCCCCCC1=CC(=O)C2=CC=CC=C2N1C"])
# res_fp = fpr.get_fp(res[0]["fingerprint"])
# res_fp[0,169] 


if __name__ == '__main__':
    unittest.main()
