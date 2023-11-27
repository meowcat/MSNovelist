
from tqdm import tqdm
import sys
sys.path.append('/msnovelist')

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import smiles_config as sc

import infrastructure.generator as gen

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, \
    LambdaCallback, Callback

import numpy as np
import pandas as pd
import time
import math
import os
import pickle
import scipy

# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("testbed startup")

import importlib

import seaborn as sns



pipeline_x = sc.config['pipeline_x']
pipeline_y = sc.config['pipeline_y']
logger.info(f"pipeline_x: {pipeline_x}")
logger.info(f"pipeline_y: {pipeline_y}")

training_id = str(int(time.time()))
if sc.config['training_id'] != '':
    training_id = sc.config['training_id']

sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
training_set = f"fold[^{cv_fold}]"
validation_set = 'fold0'
if cv_fold != 'X':
    validation_set = f"fold{cv_fold}"



# sampler_name = sc.config['sampler_name']
sampler_name = "bitmatrix"
sampler_module = None
if sampler_name != '':
    sampler_module = importlib.import_module('fp_sampling.' + sampler_name, 'fp_sampling')
#import models.quicktrain_fw_20190327 as sm


data_eval_ =  sc.config["db_path_eval"]
# note: with CV, the evaluation set name is the same as the validation set name
db_eval = db.FpDatabase.load_from_config(data_eval_)
dataset_eval = db_eval.get_grp(validation_set)
logger.info(f"Datasets - building pipeline for evaluation")
fp_dataset_eval_ = gen.smiles_pipeline(dataset_eval, 
                                    batch_size = sc.config['batch_size'],
                                    map_fingerprints=False,
                                    **db_eval.get_pipeline_options())

logger.info(f"Datasets - pipelines built")

# If fingerprint sampling is configured: load the sampler and map it
if sampler_module is not None:
    logger.info(f"Sampler {sampler_name} loading")
    sampler_factory = sampler_module.SamplerFactory(sc.config)
    sampler = sampler_factory.get_sampler()
    logger.info(f"Sampler {sampler_name} loaded")
    fp_dataset_eval_ = sampler.map_dataset(fp_dataset_eval_)

dataset = tf.data.Dataset.zip(fp_dataset_eval_)

data = next(iter(dataset))

def tanimoto(fp1, fp2):
    '''
    Calculates the tanimoto similarity between two equally-sized blocks
    of fingerprints, with samples in rows.
            
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
    
    fp_1 = tf.cast(fp1, 'uint8')
    fp_2 = tf.cast(fp2, 'uint8')

    fp11 = fp_1 * fp_2
    fp10 = fp_1 * (1-fp_2)
    #fp00 = (1-fp_1) *(1-fp_2)
    fp01 = (1-fp_1) * fp_2
    
    c11 = tf.cast(tf.reduce_sum(fp11, axis=1), 'float')
    c10 = tf.cast(tf.reduce_sum(fp10, axis=1), 'float')
    c01 = tf.cast(tf.reduce_sum(fp01, axis=1), 'float')

    tanimoto_ = c11 / (c11 + alpha * c10 + beta * c01)
    return tf.cast(tanimoto_, "float32")
        
data_iter = iter(data)

tanimoto_dist_predicted = []
tanimoto_dist_sampled = []
for _ in range(20):
    tanimoto_dist_predicted_ = tanimoto(data["fingerprint"], tf.round(data["fingerprint_degraded"]))
    tanimoto_dist_sampled_ = tanimoto(data["fingerprint"], tf.round(data["fingerprint_sampled"]))
    tanimoto_dist_predicted.extend(tanimoto_dist_predicted_.numpy())
    tanimoto_dist_sampled.extend(tanimoto_dist_sampled_.numpy())


wasserstein = scipy.stats.wasserstein_distance(
    tanimoto_dist_predicted, 
    tanimoto_dist_sampled
    )

sns.histplot({
    'predicted': tanimoto_dist_predicted,
     'sampled': tanimoto_dist_sampled
     }).set(title = wasserstein)


