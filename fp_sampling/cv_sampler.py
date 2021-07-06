# -*- coding: utf-8 -*-

from fp_sampling.sampling import *
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from fp_management import database as db

import pandas as pd
import numpy as np
import tensorflow as tf

import random

class SamplerFactory:
    def __init__(self, config):
        self.base_folder = config["base_folder"]
        self.fp_map_ = config["fp_map"]
        self.loaded = False
        self.cv_folds = config["cv_folds"]
        self.cv_fold = config["cv_fold"]
        self.db_path_eval = config["db_path_eval"]
        self.validation_set= config["validation_set"]
        
    def _load(self):
        # Load mapping table for the CSI:FingerID predictors
        self.fp_map = fpm.FingerprintMap(self.fp_map_)
        # Currently, we subset the map to 3541 bits so it matches
        # the layout of crossvalidation_open as it is now
        
        
        
        db_path_sample = {
            'path': 'C:\\Daten\\Unicorn\\Data\\evaluation_v44/dataset2/predictions_smiles.csv',
            'fp_map': 'C:\\Daten\\Unicorn\\Data\\sirius-4.4.17\\csi_fingerid.csv',
            'construct_from': 'smiles',
            'reload_smiles_pubchem': True}
        
        # Load both the evaluation dataset (to exclude from sampling)
        # and the database to sample from 
        
        self.db_sample = db.FpDatabase.load_from_config(db_path_sample)
        
        
        if self.cv_folds == 1:
            db_path_eval = self.db_path_eval
            self.db_eval = db.FpDatabase.load_from_config(db_path_eval)
            eval_grp = self.db_eval.get_grp(self.validation_set)
            eval_inchikey1 = set([entry["inchikey1"] for entry in eval_grp])
        else:
            raise NotImplementedError("CV is still WIP!")
        
        self.eval_inchikey1 = eval_inchikey1
        
        self.loaded = True
        
    
    def get_sampler(self):

        if not self.loaded:
            self._load()
        
        db_sample_select = self.db_sample.data_information.loc[
            ~self.db_sample.data_information["inchikey1"].isin(self.eval_inchikey1)
            ]
        
        fp_map = self.fp_map.positions
        
        fp_true = self.db_sample.data_fp_true[db_sample_select.row_id,:][:,fp_map]
        fp_predicted = self.db_sample.data_fp_predicted[db_sample_select.row_id,:][:,fp_map]
        
        sampler = CVSampler(
            fp_true,
            fp_predicted,
            noise = 0.15)
        return sampler
    
    def round_fingerprint_inference(self):
        return False

        

        
        


class CVSampler(Sampler):
    def __init__(self, fp_true, fp_predicted, noise,
                 generator = tf.random.experimental.get_global_generator()):
        '''
        Initializes a random probabilistic sampler, i.e. which samples 
        simulated probabilistic fingerprints (with some added noise) for a true 
        fingerprint, randomly - i.e. picking a random prediction for every
        bit of corresponding value.
        
        Parameters
        ----------
        fp_true: np.array (n, fp_len)
            Array of n true fingerprints, expected to be € {0, 1}
        fp_predicted:
            Array of n CSI:FingerID crossvalidated predictions,
            expected to be € [0, 1] (or really, any value, this does not matter)
        noise:
            A noise scaling factor for adding uniform noise to the fingerprint.
        generator: tf.random.Generator
        Returns
        -------
        None.

        '''
        Sampler.__init__(self)
        
        # Append a completely 1 and a completely 0 fingerprint to have
        # at least one value per bit
        fp_true = np.concatenate([
                fp_true,
                np.zeros_like(fp_true[:1,:]),
                np.ones_like(fp_true[:1,:]),
            ], axis=0)
        
        # Note: This throws out quite a few fingerprint bits.
        # 95 bits have zero positive examples in this dataset!
        # Perhaps a better idea would be to use the stats as a fallback.

        # At this value, add a complete misprediction so we don't mistakenly
        # teach the network that this prediction is "good" - if we set 0.5 here,
        # this would mean that if all fp_predicted < 0.1 for fp_true = 0,
        # fp_predicted = 0.5 uniquely identifies fp_true = 1.
        # Anecdotally, for the 95 bits, 100 sampled bits0 were was below 0.1!
        fp_predicted = np.concatenate([
                fp_predicted,
                np.ones_like(fp_true[:1,:]),
                np.zeros_like(fp_true[:1,:]),
            ], axis=0)
        
        self.fp_true = fp_true.astype("float32")
        self.fp_predicted = fp_predicted.astype("float32")
        self.noise = noise
        self.generator = generator
        
        # Initialize the blocks with indexers for where 1 and 0 bits are
       
        fp_1_where = np.where(self.fp_true.transpose() == 1)
        self.fp_1_pos = fp_1_where[1].astype("int32")
        self.fp_1_sum = np.sum(self.fp_true == 1, axis=0).astype("int32")
        fp_1_cumsum = np.hstack([[0], self.fp_1_sum.copy()])
        fp_1_cumsum.cumsum(out=fp_1_cumsum)
        self.fp_1_cumsum = fp_1_cumsum[:-1].astype("int32")

        fp_0_where = np.where(self.fp_true.transpose() == 0)
        self.fp_0_pos = fp_0_where[1].astype("int32")
        self.fp_0_sum = np.sum(self.fp_true == 0, axis=0).astype("int32")
        fp_0_cumsum = np.hstack([[0], self.fp_0_sum.copy()])
        fp_0_cumsum.cumsum(out=fp_0_cumsum)
        self.fp_0_cumsum = fp_0_cumsum[:-1].astype("int32")

    @tf.function
    def sample(self, fp, test_lookup=False):
        '''
        Generates simulated predicted fingerprints Y for an array of true 
        fingerprints X [x_j, i] of shape (n, fp_len)
        (i.e. i is the bit in the fingerprint, j is the fingerprint in the batch)
        
        using probabilistic random sampling from the loaded 
        set of predicted fingerprints.

        Parameters
        ----------
        fp : np.array of tf.Tensor(shape=(n, fp_len)) dtype=float32 but 
            Array of n fingerprints expected to be € {0, 1} (but of float type)

        Returns
        -------
        Equally-shaped tensor with simulated predicted probabilistic fingerprint, i.e. 
            all values are € (0, 1]

        '''
        
        rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
        # #self.fp_0_pos[
        # source_choice_0 = tf.gather(
        #     self.fp_0_pos,
        #     rand % self.fp_0_sum + self.fp_0_cumsum)
        # source_choice_1 = tf.gather(
        #     self.fp_1_pos,
        #     rand % self.fp_1_sum + self.fp_1_cumsum)
        
        
        
             
        source_choice_0 = tf.gather(
            self.fp_0_pos,
            rand % tf.expand_dims(self.fp_0_sum, 0)
                + tf.expand_dims(self.fp_0_cumsum, 0))
        source_choice_1 = tf.gather(
            self.fp_1_pos,
            rand % tf.expand_dims(self.fp_1_sum, 0)
                + tf.expand_dims(self.fp_1_cumsum, 0))
        
        source_map = tf.repeat(tf.expand_dims(
            tf.range(tf.shape(source_choice_0)[1]),
            0), tf.shape(source_choice_0)[0], axis=0)
        
        source_map_0 = tf.transpose(
            tf.stack([source_choice_0, source_map]),
            [1,2,0])
        source_map_1 = tf.transpose(
            tf.stack([source_choice_1, source_map]),
            [1,2,0])
        
        bits_0 = tf.gather_nd(self.fp_predicted, source_map_0, batch_dims=0)
        bits_1 = tf.gather_nd(self.fp_predicted, source_map_1, batch_dims=0)
        # The distributions here look right (histogram),
        # but I have no unit test yet 
        # for this last line below...
        fp_simulated = fp * bits_1 + (1-fp) * bits_0
        fp_noise = self.noise * (self.generator.uniform(tf.shape(fp_simulated)) - 0.5)
        fp_simulated = tf.clip_by_value(fp_simulated + fp_noise, 
                                        tf.reduce_min(fp_simulated),
                                        tf.reduce_max(fp_simulated))

        if test_lookup:
            test = tf.gather_nd(self.fp_true, source_map_0, batch_dims=0)
            fail_count_0 = tf.reduce_sum(tf.cast(test > 0, "int32"))
            # raise ValueError("Lookup returned non-zeros in reference fingerprint for zero positions")
            test = tf.gather_nd(self.fp_true, source_map_1, batch_dims=0)
            fail_count_1 = tf.reduce_sum(tf.cast(test < 1, "int32"))
            return fp_simulated, fail_count_0, fail_count_1

        return fp_simulated
        
        
        
        