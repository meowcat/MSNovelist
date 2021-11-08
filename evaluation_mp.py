# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:30 2020

@author: stravsm
"""

import importlib
from importlib import reload
from tqdm import tqdm
import os

import tensorflow as tf
import numpy as np
import pandas as pd

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import smiles_config as sc
import infrastructure.generator as gen
import infrastructure.decoder as dec

import time
from datetime import datetime
import pickle


from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import infrastructure.score as msc
import gc
import random

# Randomness is relevant for stochastic sampling
random_seed = sc.config['random_seed_global']
if random_seed != '':
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.experimental.set_seed(random_seed)

# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("evaluation_mp startup")

eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
    
if isinstance(sc.config['weights'], list):
    weights_list = sc.config['weights']
else:
    weights_list = [sc.config['weights']]
    

# First, do everything independent of weights

fp_map = fpm.FingerprintMap(sc.config["fp_map"])
fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = False,
                                  cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()


    
n = sc.config["eval_n"]
n_total = sc.config["eval_n_total"]
#n_total_ = n_total // n * n
k = sc.config["eval_k"]
kk = sc.config["eval_kk"]
steps = sc.config["eval_steps"]

decoder_name = sc.config["decoder_name"]

sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
evaluation_set_ = sc.config['evaluation_set']
evaluation_set = f"fold{cv_fold}-{evaluation_set_}"

# File for CSI:FingerID validation data
data_eval_ = sc.config["db_path_eval"]
# Load mapping table for the CSI:FingerID predictors
# Load dataset and process appropriately
db_eval = db.FpDatabase.load_from_config(data_eval_)
pipeline_options =  db_eval.get_pipeline_options()
    
pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']

dataset_val = db_eval.get_grp(evaluation_set)
if n_total != -1:
    dataset_val = dataset_val[:n_total]
else:
    n_total = len(dataset_val)

# Load dataset and sampler, apply sampler to dataset
# (so we can also evaluate from fingerprint_sampled)
fp_dataset_val_ = gen.smiles_pipeline(dataset_val, 
                                    batch_size = n,
                                    fp_map = fp_map.positions,
                                    **pipeline_options)

fp_dataset_val = gen.dataset_zip(fp_dataset_val_, 
                                 pipeline_encoder, pipeline_reference,
                                 **pipeline_options)

sampler_name = sc.config['sampler_name']
round_fingerprints = True
if sampler_name != '':
    logger.info(f"Sampler {sampler_name} loading")
    spl = importlib.import_module(sampler_name, 'fp_sampling')
    sf = spl.SamplerFactory(sc.config)
    round_fingerprints = sf.round_fingerprint_inference()
    sampler = sf.get_sampler()
    logger.info(f"Sampler {sampler_name} loaded")
    fp_dataset_val_ = sampler.map_dataset(fp_dataset_val_)


for weights_i, weights_ in enumerate(weights_list):
    eval_id = str(int(time.time()))
    pickle_id = eval_id
    if sc.config['eval_id'] != '':
        eval_id = sc.config['eval_id']
    if sc.config['eval_counter'] != '':
        pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
        if len(weights_list) > 1:
            pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter'] + "-" + weights_i
        
    logpath_topn = sc.config["eval_folder"] + "eval_" + eval_id + "_topn.txt"
    logpath_top1 = sc.config["eval_folder"] + "eval_" + eval_id + "_top1.txt"
    picklepath = sc.config["eval_folder"] + "eval_" + pickle_id + ".pkl"
    logger.info(picklepath)
    logger.info(weights_)
    weights = os.path.join(sc.config["weights_folder"], weights_)

    
    retain_single_duplicate = True

    fp_dataset_iter = iter(fp_dataset_val)
    blueprints = gen.dataset_blueprint(fp_dataset_val_)
    
    # Load models
    
    import model
    
    model_encode = model.EncoderModel(
                     blueprints = blueprints,
                     config = sc.config,
                     round_fingerprints = round_fingerprints)
    model_decode = model.DecoderModel(
                     blueprints = blueprints,
                     config = sc.config,)
    model_transcode = model.TranscoderModel(
                    blueprints = blueprints,
                     config = sc.config,
                     round_fingerprints = round_fingerprints)
    
    # Build models by calling them
    y_ = model_transcode(blueprints)
    enc = model_encode(next(fp_dataset_iter)[0])
    _ = model_decode(enc)
    
    model_transcode.load_weights(weights, by_name=True)
    model_encode.copy_weights(model_transcode)
    model_decode.copy_weights(model_transcode)
    
    
    # Initialize decoder
    decoder = dec.get_decoder(decoder_name)(
        model_encode, model_decode, steps, n, k, kk, config = sc.config)
    logger.info("Decoder initialized")
    logger.info(f"Processing and scoring predictions")
    
    logger.info(f"Predicting {n_total} samples - start")
    logger.info(f"Beam block size {n}*{k}*{steps}, sequences retrieved per sample: {kk}")
    result_blocks = []
    reference_blocks = []
    for data in tqdm(fp_dataset_val, total = (n_total -1) // n + 1):
        # repeat the input data k times for each of n queries
        # (now we encode each of k samples individually because the encoding
        # may be probabilistic)
        
        # make a custom decoder if we don't have all n samples
        n_real = len(data[0]['n_hydrogen'])
        if n_real != n:
            decoder = dec.get_decoder(decoder_name)(
                    model_encode, model_decode, steps, n_real, k, kk, config = sc.config)
        
        data_k = {key: tf.repeat(x, k, axis=0) for key, x in data[0].items()}
        states_init = model_encode.predict(data_k)
        # predict k sequences for each query.
        sequences, y, scores = decoder.decode_beam(states_init)
        seq, score, length = decoder.beam_traceback(sequences, y, scores)
        smiles = decoder.sequence_ytoc(seq)
        results_df = decoder.format_results(smiles, score)
        result_blocks.append(results_df)
        reference_df = decoder.format_reference(
            [bytes.decode(x, 'UTF-8') for x in data[1][0].numpy()],
            [d for d in data[1][1].numpy()])
        reference_blocks.append(reference_df)
    results = pd.concat(result_blocks)        
    logger.info(f"Predicting {n_total} samples - done")
    
    logger.info(f"Evaluating {n_total} blocks - start")
    
    results_evaluated = []
    for block_, ref_, block_id in zip(tqdm(result_blocks), 
                                    reference_blocks,
                                    range(len(result_blocks))):
        # Make a block with molecule, MF, smiles for candidates and reference
        block = db.process_df(block_, fingerprinter,
                              construct_from = "smiles",
                              block_id = block_id)
        
        if retain_single_duplicate:
            block.sort_values("score", ascending = False, inplace = True)
            block = block.groupby(["n", "inchikey1"]).first().reset_index()
            
        ref = db.process_df(ref_, fingerprinter,
                              construct_from = "smiles",
                              block_id = block_id)
        # Also actually compute the true fingerprint for the reference
        fingerprinter.process_df(ref,
                                 out_column = "fingerprint_ref_true",
                                 inplace=True)
        
        # Match ref to predictions
        block = block.join(ref, on="n", rsuffix="_ref")
        # Keep only correct formula
        block_ok = block.loc[block["inchikey1"].notna()].loc[block["mf"] == block["mf_ref"]]
        # Now actually compute the fingerprints, only for matching MF
        fingerprinter.process_df(block_ok,
                                 inplace=True)
        block = block.merge(
            block_ok[["n","k","fingerprint"]],
            left_on = ["n", "k"],
            right_on = ["n", "k"],
            suffixes = ["_ref", ""],
            how = "left")
    
        results_evaluated.append(block)
        
    logger.info(f"Evaluating {n_total} blocks - merging")
    results_complete = pd.concat(results_evaluated)
    results_complete["nn"] = n * results_complete["block_id"] + results_complete["n"]
    results_complete ["evaluation_set"] = evaluation_set
    
    logger.info(f"Pickling predictions from [{evaluation_set}]")
    pickle.dump(results_complete, open(picklepath, "wb"))
    
    results_ok = results_complete.loc[results_complete["fingerprint"].notna()].copy()
