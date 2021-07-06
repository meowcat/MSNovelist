# -*- coding: utf-8 -*-

from fp_sampling.sampling import *
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from fp_management import database as db

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import math as tm

import warnings

import random

class SamplerFactory:
    def __init__(self, config):
        self.base_folder = config["base_folder"]
        self.fp_map_ = config["fp_map"]
        self.loaded = False
        self.cv_folds = config["cv_folds"]
        self.cv_fold = config["cv_fold"]
        self.db_path_sample = config["db_path_sample"]
        self.sampler_config = config["sampler_config"]
        self.round_fingerprint = False
        if 'final_round' in self.sampler_config.keys():
            self.round_fingerprint = self.sampler_config['final_round']
        
    def _load(self):
        # Load mapping table for the CSI:FingerID predictors
        self.fp_map = fpm.FingerprintMap(self.fp_map_)
        # Currently, we subset the map to 3541 bits so it matches
        # the layout of crossvalidation_open as it is now
        
        # Load both the evaluation dataset (to exclude from sampling)
        # and the database to sample from 
        
        self.db_sample = db.FpDatabase.load_from_config(self.db_path_sample)
        
        self.loaded = True
        
    
    def get_sampler(self):

        if not self.loaded:
            self._load()
        
        # Remove candidates from inactive fold
        db_sample_select = self.db_sample.data_information.loc[
            ~self.db_sample.data_information.index.str.startswith(f"fold{self.cv_fold}-")
            ]
        
        fp_map = self.fp_map.positions
        
        fp_true = self.db_sample.data_fp_true[db_sample_select.row_id,:][:,fp_map]
        fp_predicted = self.db_sample.data_fp_predicted[db_sample_select.row_id,:][:,fp_map]
        
        sampler = CVSampler(
            fp_true,
            fp_predicted,
            self.sampler_config)
        return sampler
    
    def round_fingerprint_inference(self):
        return self.round_fingerprint

        

        
        


class CVSampler(Sampler):
    def __init__(self, fp_true, fp_predicted, sampler_config,
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
        
        # Sampler configuration. Default plus settings
        sampler_config_ = {
                'use_similar': False,
                'n_top': 0,
                'max_loop': 10,
                'mask_random': 0,
                'replace_random': 0,
                'replace_true': 0,
                'fill_remaining': True,
                'final_jitter': 0.1,
                'final_round': False,
            }
        sampler_config_.update(sampler_config)
        self.use_similar = tf.convert_to_tensor(sampler_config_['use_similar'], "bool")
        self.n_top = tf.convert_to_tensor(sampler_config_['n_top'], "int32")
        self.max_loop = tf.convert_to_tensor(sampler_config_['max_loop'], "int32")
        self.mask_random = tf.convert_to_tensor(sampler_config_['mask_random'], "float32")
        self.replace_random = tf.convert_to_tensor(sampler_config_['replace_random'], "float32")
        self.replace_true = tf.convert_to_tensor(sampler_config_['replace_true'], "float32")
        self.fill_remaining = tf.convert_to_tensor(sampler_config_['fill_remaining'], "bool")
        self.final_jitter = tf.convert_to_tensor(sampler_config_['final_jitter'], "float32")
        self.final_round = tf.convert_to_tensor(sampler_config_['final_round'], "bool")
        
        if self.use_similar:
            warnings.warn("use_similar is currently deactivated to check if optimization is influenced")
        
        # Append one completely TP, FP, TN, FN fingerprint
        # to have at least one sample per bit and type
        # at least one value per bit
        fp_true = np.concatenate([
                fp_true,
                np.zeros_like(fp_true[:1,:]),
                np.ones_like(fp_true[:1,:]),
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
                np.ones_like(fp_true[:1,:]),
                np.zeros_like(fp_true[:1,:]),
                np.zeros_like(fp_true[:1,:])
            ], axis=0)
        
        self.fp_true = tf.convert_to_tensor(fp_true, "float32")
        self.fp_predicted = tf.convert_to_tensor(fp_predicted, "float32")
        self.generator = generator
        
        # Initialize the blocks with indexers for where 1 and 0 bits are
        fp_TP_where = tf.transpose((self.fp_true == 1) & (self.fp_predicted > 0.5))
        fp_FP_where = tf.transpose((self.fp_true == 0) & (self.fp_predicted > 0.5))
        fp_FN_where = tf.transpose((self.fp_true == 1) & (self.fp_predicted < 0.5))
        fp_TN_where = tf.transpose((self.fp_true == 0) & (self.fp_predicted < 0.5))

        self.fp_where = tf.cast(
            tf.stack([fp_TP_where, fp_FP_where, fp_FN_where, fp_TN_where]), "float32")
        self.fp_positions = tf.where(self.fp_where)
        self.fp_sums = tf.cast(tf.reduce_sum(self.fp_where, axis=2), "int32")
        fp_cumsums = tf.cumsum(
            tf.reshape(self.fp_sums, [-1]), exclusive = True)
        self.fp_cumsums = tf.reshape(fp_cumsums,[4,-1])
        self.fp_values = tf.gather_nd(
            tf.transpose(self.fp_predicted),
            self.fp_positions[:,1:]
            )
        
        self.fp_where_uncorrelated = tf.cast(
            tf.stack([fp_TP_where | fp_FN_where,
                      fp_TN_where | fp_FP_where]), "float32")
        self.fp_positions_uncorrelated = tf.where(self.fp_where_uncorrelated)
        self.fp_sums_uncorrelated = tf.cast(tf.reduce_sum(self.fp_where_uncorrelated, axis=2), "int32")
        fp_cumsums_uncorrelated = tf.cumsum(
            tf.reshape(self.fp_sums_uncorrelated, [-1]), exclusive = True)
        self.fp_cumsums_uncorrelated = tf.reshape(fp_cumsums_uncorrelated,[2,-1])
        self.fp_values_uncorrelated = tf.gather_nd(
            tf.transpose(self.fp_predicted),
            self.fp_positions_uncorrelated[:,1:]
            )
        
        
    #@tf.function
    def partial_tanimoto_ref(self, fp, positions_):
        '''
        Calculates the tanimoto similarity between n fingerprints in fp
        and m reference fingerprints in self.fp_true, for all positions
        specified in positions.
        
        Correspondingly, returns a tensor nxm of float 0..1.
                
        '''
        # Some code from:
        # https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
        # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
        # -> the score is computed for each class separately and then summed
        # alpha=beta=0.5 : dice coefficient
        # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
        # alpha+beta=1   : produces set of F*-scores
        # implemented by E. Moebel, 06/04/18
        
        alpha = 1
        beta = 1
        
        fp_subset = tf.cast(fp, 'uint8')
        ref_subset = tf.cast(self.fp_true, 'uint8')
        positions = tf.cast(positions_, 'uint8')
        ones_fp = tf.ones_like(fp_subset)
        ones_ref = tf.ones_like(ref_subset)
        
        fp_1 = tf.expand_dims(positions * fp_subset, 1)
        fp_0 = tf.expand_dims(positions * ones_fp - fp_subset, 1)
        ref_1 = tf.expand_dims(ref_subset, 0)
        ref_0 = tf.expand_dims(ones_ref - ref_subset, 0)
        
        # Tanimoto is C11 / (C10+C01+C11) but only unfilled positions are counted
        c11 = tf.reduce_sum(fp_1 * ref_1, axis=2)
        c10 = tf.reduce_sum(fp_1 * ref_0, axis=2)
        c01 = tf.reduce_sum(fp_0 * ref_1, axis=2)
        tanimoto = c11 / (c11 + alpha * c10 + beta * c01)
        return tf.cast(tanimoto, "float32")
        
               
    @tf.function
    def sample(self, fp):
        fp_simulated, _ = self.sample_(fp)
        return fp_simulated
    
    @tf.function
    def sample_with_missing(self, fp):
        fp_simulated, missing = self.sample_(fp)
        return fp_simulated, missing

    @tf.function
    def sample_step(self, fp, fp_simulated, fp_positions_empty, 
                    fingerprint_remaining, steps):
        # Two methods:
        # Either sample from very similar fingerprints
        # Or sample from totally random fingerprints

        fp_selected = self.generator.uniform_full_int((tf.shape(fp)[0],), dtype='int32') % tf.shape(self.fp_true)[0]
        fp_sample = tf.gather(self.fp_true, fp_selected)
        
        fp_value_to_sample = tf.cast(tf.gather(self.fp_predicted, fp_selected) > 0.5, 'float32')
        # find the positions to sample:
        # 1. 
        # Starting from "all unsampled positions", put a random mask to limit
        # sampling more than "random_mask %" from a single fingerprint
        # (This is not in the CANOPUS version)
        rand = tf.cast(self.generator.uniform(tf.shape(fp), dtype="float32") > self.mask_random, "float32")
        fp_positions_sampling = rand * fp_positions_empty
        # 2. 
        # For the non-random-masked unsampled positions, find bits that match
        # between the query and the chosen library fingerprint.
        # For every 1-1 match, 
        # sample a true positive if the simulated library fingerprint is a true positive; 
        # and a false negative if the simulated library fingerprint is a false negative.
        c11 = tf.cast(fp * fp_sample, "float32") * fp_positions_sampling
        sample_tp = c11 * fp_value_to_sample
        sample_fn = c11 * (1-fp_value_to_sample)
        c00 = tf.cast((1-fp) * (1-fp_sample), "float32") * fp_positions_sampling
        sample_tn = c00 * (1-fp_value_to_sample)
        sample_fp = c00 * (fp_value_to_sample)
        # 3. 
        # This is perhaps the hardest to understand from the code:
        # Choose the position within the array of TP,FP,TN,FN values
        # Let TP=0,FP=1,FN=2,TN=3 be called the four "strata" to choose from
        # for each bit. 
        # fp_sums is the number of entries for each bit and stratum, and
        # fp_cumsums is the starting point for each bit and stratum.
        # So chosen_position will be a n_query * 4 (strata) * n_bits tensor
        # of positions in an 1-d array (in which the values for each 
        # bit and stratum are stored)
        rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
        rand = tf.expand_dims(rand, 1)
        chosen_position = (rand % tf.expand_dims(self.fp_sums, 0) 
                           + tf.expand_dims(self.fp_cumsums, 0) )
        # 4.
        # Extract the value from the 1-d array (this is the easy part :) )
        sampled_bit = tf.gather(self.fp_values, chosen_position)
        # 5. 
        # We now have a value to sample for every bit AND stratum of the
        # query fingerprints. By stacking together the "stratum choice" and
        # multiplying, then summing the stratum values for each bit,
        # we find which value to actually add to the fingerprint.
        # Reminder: Let TP=0,FP=1,FN=2,TN=3
        chosen_stratum = tf.stack([sample_tp, sample_fp, sample_fn, sample_tn], axis=1)            
        chosen_bit = tf.reduce_sum(
            sampled_bit * tf.cast(chosen_stratum, "float32"), axis = 1)
        # 6.            
        # Add the sampled bits to the simulated fingerprint
        # (Note: the (c11+c00) should be unnecessary, since this is already
        # calculated out in step 2)
        fp_simulated = fp_simulated + (chosen_bit * (c11 + c00))
        # 7.
        # Remove the sampled positions from the tensor of unsampled positions
        fp_positions_empty = fp_positions_empty - c11 - c00
        
        steps = steps + 1
        return fp, fp_simulated, fp_positions_empty, fingerprint_remaining, steps

    @tf.function
    def sample_remaining(self, fp, fp_simulated, fp_positions_empty, replace = 0):
        # Fill the remaining positions with random, uncorrelated sampling 
        # from TP + FN or TN + FP

        # Additionally, replace some of the correlatedly sampled positions
        # with random samples
        replace_positions = self.generator.uniform(tf.shape(fp), dtype="float32")
        replace_positions = tf.cast(replace_positions < replace, "float32")
        replace_positions = replace_positions * (1-fp_positions_empty)
        
        fill_positions = (fp_positions_empty + replace_positions)
        
        c1 = fp * fill_positions
        c0 = (1-fp) * fill_positions
        
        rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
        rand = tf.expand_dims(rand, 1)
        # Position chosen for each of c1, c0:
        
        chosen_position = (rand % tf.expand_dims(self.fp_sums_uncorrelated, 0) 
                           + tf.expand_dims(self.fp_cumsums_uncorrelated, 0) )
        sampled_bit = tf.gather(self.fp_values_uncorrelated, chosen_position)
        # Here:
        # TP+FN = c1
        # TN+FP = c0
        chosen_stratum = tf.stack([c1, c0], axis=1)
        
        chosen_bit = tf.reduce_sum(
            sampled_bit * tf.cast(chosen_stratum, "float32"), axis = 1)
        
        # Compose the final fingerprint from the previous simulated FP
        # and the newly sampled positions
        fp_simulated = (fp_simulated * (1-fill_positions)) + (chosen_bit * fill_positions)
        
        return fp_simulated
    
    @tf.function
    def sample_true(self, fp, fp_simulated, replace = 0):
        '''
        Replace a subset of the fingerprint with only correct predictions
        '''
        replace_score = tf.expand_dims(
            self.generator.uniform((tf.shape(fp)[0],)),
            1
            ) + 2*(replace-0.5)
        
        replace_positions = self.generator.uniform(tf.shape(fp), dtype="float32")
        replace_positions = tf.cast(replace_positions < replace_score, "float32")
        sample_tp =  fp * replace_positions
        sample_tn = (1-fp) * replace_positions
        sample_fn = tf.zeros_like(sample_tn)
        sample_fp = tf.zeros_like(sample_tn)
        
        rand = self.generator.uniform_full_int(tf.shape(fp), dtype="int32")
        rand = tf.expand_dims(rand, 1)
        chosen_position = (rand % tf.expand_dims(self.fp_sums, 0) 
                           + tf.expand_dims(self.fp_cumsums, 0) )
        
        # Extract the value from the 1-d array (this is the easy part :) )
        sampled_bit = tf.gather(self.fp_values, chosen_position)
        # 
        # We now have a value to sample for every bit AND stratum of the
        # query fingerprints. By stacking together the "stratum choice" and
        # multiplying, then summing the stratum values for each bit,
        # we find which value to actually add to the fingerprint.
        # Reminder: Let TP=0,FP=1,FN=2,TN=3
        chosen_stratum = tf.stack([sample_tp, sample_fp, sample_fn, sample_tn], axis=1)            
        chosen_bit = tf.reduce_sum(
            sampled_bit * tf.cast(chosen_stratum, "float32"), axis = 1)
        # 6.            
        # Add the sampled bits to the simulated fingerprint
        # (Note: the (c11+c00) should be unnecessary, since this is already
        # calculated out in step 2)
        fp_simulated = ((fp_simulated * (1-replace_positions)) + 
                        (chosen_bit * replace_positions))
        # 7.
        # Remove the sampled positions from the tensor of unsampled positions
        return fp_simulated
        
    @tf.function
    def sample_(self, fp):

        '''
            def sample_(self, fp, 
                max_loop = 5, 
                use_similar = False, 
                n_top = 10,
                mask_random = 0,
                replace_random = 0, 
                fill_remaining = True,
                final_jitter = 0.1,
                final_round = False
                ):
        
        
        Generates simulated predicted fingerprints Y for an array of true 
        fingerprints X [x_j, i] of shape (n, fp_len)
        (i.e. i is the bit in the fingerprint, j is the fingerprint in the batch)
        
        using probabilistic correlated sampling from the loaded 
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
        max_loop = self.max_loop
        n_top = self.n_top
        mask_random = self.mask_random
        replace_random = self.replace_random
        fill_remaining = self.fill_remaining
        final_jitter = self.final_jitter
        final_round = self.final_round
        replace_true = self.replace_true

        
        
        
        fp_positions_empty = tf.ones_like(fp, dtype="float32")
        # fingerprint_remaining = tf.ones((fp.shape[0],self.fp_true.shape[0])).numpy()
        # fingerprint_remaining[0,4068] = 0
        # fingerprint_remaining[1,8589] = 0
        # fingerprint_remaining = tf.convert_to_tensor(fingerprint_remaining)
        fingerprint_remaining = tf.ones((tf.shape(fp)[0],tf.shape(self.fp_true)[0]), dtype="float32")
        fp = tf.cast(fp, "float32")
        
        fp_simulated = tf.zeros_like(fp)
        
        def sample_body(fp, fp_simulated, fp_positions_empty, 
                             fingerprint_remaining, steps):
            return self.sample_step(fp, fp_simulated,
                                    fp_positions_empty,
                                    fingerprint_remaining,
                                    steps)
        # Condition for the loop
        # Todo: check if it goes faster if we just loop n times
        # regardless of the need
        def sample_condition(fp, fp_simulated, fp_positions_empty, 
                             fingerprint_remaining, steps):
            return steps < max_loop
            #return tf.logical_and(tf.reduce_sum(fp_positions_empty) > 0, 
            #                      steps < max_loop)
        
        # Run the sampling procedure for a maximum of max_loop iterations
        steps = 0
        _, fp_simulated, fp_positions_empty, _, steps = tf.while_loop(
            sample_condition,
            sample_body,
            [fp, fp_simulated, fp_positions_empty, fingerprint_remaining, steps]
            )            
        
        # Fill all still unfilled positions if fill_remaining is set
        fp_simulated = tf.cond(
            fill_remaining,
            lambda: self.sample_remaining(fp, fp_simulated, fp_positions_empty, replace_random),
            lambda: fp_simulated
            )
        
        fp_simulated = self.sample_true(fp, fp_simulated, self.replace_true)
        
        
        fp_noise = final_jitter * (self.generator.uniform(tf.shape(fp_simulated)) - 0.5)
        fp_simulated = tf.clip_by_value(fp_simulated + fp_noise, 
                                        tf.reduce_min(fp_simulated),
                                        tf.reduce_max(fp_simulated))
        
        # Round the fingerprint if final_round is set
        fp_simulated = tf.cond(
            final_round,
            lambda: tf.round(fp_simulated),
            lambda: fp_simulated
            )


        return fp_simulated, fp_positions_empty
        
        
        
        