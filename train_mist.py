#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import sys
sys.path.append('/home/stravsmi/msmsgym/MSNovelist-private')


from fp_management import database as db
from fp_management import mist_fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import os


# In[2]:




import smiles_process as sp
import importlib
from importlib import reload
import smiles_config as sc

sc.config_file.append('config.EULER.yaml')
sc.config_reload()

import infrastructure.generator as gen

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard,     LambdaCallback, Callback

import numpy as np
import pandas as pd
import time
import math
import pickle
import json


# In[3]:





# In[4]:



# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("training startup")


# In[5]:



sampler_name = sc.config['sampler_name']
sampler_module = None
if sampler_name != '':
    sampler_module = importlib.import_module('fp_sampling.' + sampler_name, 'fp_sampling')
#import models.quicktrain_fw_20190327 as sm

pipeline_x = sc.config['pipeline_x']
pipeline_y = sc.config['pipeline_y']
logger.info(f"pipeline_x: {pipeline_x}")
logger.info(f"pipeline_y: {pipeline_y}")

training_id = str(int(time.time()))
if sc.config['training_id'] != '':
    training_id = sc.config['training_id']

sc.config.setdefault('cv_fold', 0)
training_set = f"train"
validation_set = 'test'


# In[6]:



logger.info(f"Training model id {training_id}, training set")

model_tag_id = "m-" + training_id + "-" + sc.config['model_tag']
logger.info(f"Tag: {model_tag_id}")

weights_path = os.path.join(
    sc.config["weights_folder"],
    model_tag_id,
    "train")
log_path = os.path.join(
    sc.config['log_folder'],
    model_tag_id,
    "train")

config_dump_path = os.path.join(
    weights_path,
    'config.yaml'
    )

os.makedirs(weights_path)
os.makedirs(log_path)

sc.config_dump(config_dump_path)


# In[7]:


sc.config


# In[8]:


sc.config['db_path_train']


# In[9]:




logger.info(f"Datasets - loading database")
logger.info(sc.config['db_path_train'])
fp_db  = db.FpDatabase.load_from_config(sc.config['db_path_train'])
fp_train = fp_db.get_grp(training_set)
fp_val = fp_db.get_grp(validation_set)
logger.info(f"Datasets - loading evaluation")
# File for CSI:FingerID validation data




data_eval_ =  sc.config["db_path_eval"]
# note: with CV, the evaluation set name is the same as the validation set name
db_eval = db.FpDatabase.load_from_config(data_eval_)
dataset_eval = db_eval.get_grp(validation_set)


# In[10]:


fpr.MistFingerprinter.init_instance()
fingerprinter = fpr.MistFingerprinter.get_instance()


# In[14]:


logger.info(f"Datasets - building pipeline for database")


fp_dataset_train_ = gen.smiles_pipeline(fp_train, 
                                        batch_size = sc.config['batch_size'],
                                        map_fingerprints=False,
                                        **fp_db.get_pipeline_options())


# In[ ]:


next(iter(fp_dataset_train_["fingerprint"]))


# In[ ]:



fp_dataset_val_ = gen.smiles_pipeline(fp_val, 
                                        batch_size = sc.config['batch_size'],
                                        map_fingerprints=False,
                                        **fp_db.get_pipeline_options())


# In[ ]:



# logger.info(f"Datasets - building pipeline for evaluation")
# fp_dataset_eval_ = gen.smiles_pipeline(dataset_eval, 
#                                     batch_size = sc.config['batch_size'],
#                                     map_fingerprints=False,
#                                     **db_eval.get_pipeline_options())

# logger.info(f"Datasets - pipelines built")


# In[ ]:


sampler_module


# In[ ]:



# If fingerprint sampling is configured: load the sampler and map it
if sampler_module is not None:
    logger.info(f"Sampler {sampler_name} loading")
    sampler_factory = sampler_module.SamplerFactory(sc.config)
    sampler = sampler_factory.get_sampler()
    logger.info(f"Sampler {sampler_name} loaded")
    fp_dataset_train_ = sampler.map_dataset(fp_dataset_train_)
    fp_dataset_val_ = sampler.map_dataset(fp_dataset_val_)


# In[ ]:


fp1 = next(iter(fp_dataset_train_["fingerprint"]))
fp2 = next(iter(fp_dataset_train_["fingerprint_sampled"]))
fp = (fp1, fp2)


# In[ ]:


fp1.numpy().sum(axis=1), fp2.numpy().sum(axis=1)


# In[ ]:



fp_dataset_train = gen.dataset_zip(fp_dataset_train_, pipeline_x, pipeline_y,
                                   **fp_db.get_pipeline_options())
fp_dataset_train = fp_dataset_train.repeat(sc.config['epochs'])
#fp_dataset_train = fp_dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

blueprints = gen.dataset_blueprint(fp_dataset_train_)

fp_dataset_val = gen.dataset_zip(fp_dataset_val_, pipeline_x, pipeline_y,
                                 **fp_db.get_pipeline_options())
fp_dataset_val = fp_dataset_val.repeat(sc.config['epochs'])
#fp_dataset_val = fp_dataset_val.prefetch(tf.data.experimental.AUTOTUNE)

# fp_dataset_eval = gen.dataset_zip(fp_dataset_eval_, pipeline_x, pipeline_y,
#                                   **db_eval.get_pipeline_options())
# fp_dataset_eval = fp_dataset_eval.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:



training_total = len(fp_train)
validation_total= len(fp_val)
training_steps = math.floor(training_total /  sc.config['batch_size'])
if sc.config['steps_per_epoch'] > 0:
    training_steps = sc.config['steps_per_epoch']

validation_steps = math.floor(validation_total /  sc.config['batch_size'])
if sc.config['steps_per_epoch_validation'] > 0:
    validation_steps = sc.config['steps_per_epoch_validation']
    
batch_size = sc.config["batch_size"]
epochs=sc.config['epochs']

logger.info(f"Preparing training: {epochs} epochs, {training_steps} steps per epoch, batch size {batch_size}")


# In[ ]:




round_fingerprints = False
if sampler_name != '':
    round_fingerprints = sampler_factory.round_fingerprint_inference()

import model
transcoder_model = model.TranscoderModel(
    blueprints = blueprints,
    config = sc.config,
    round_fingerprints = round_fingerprints
    )

initial_epoch = 0


# In[ ]:



logger.info("Building model")

transcoder_model.compile()
#
# If set correspondingly: load weights and continue training
if 'continue_training_epoch' in sc.config: 
    if sc.config['continue_training_epoch'] > 0:
        transcoder_model.load_weights(os.path.join(
            sc.config['weights_folder'],
            sc.config['weights']))
        transcoder_model._make_train_function()
        with open(os.path.join(
                sc.config['weights_folder'],
                sc.config['weights_optimizer']), 'rb') as f:
            weight_values = pickle.load(f)
        transcoder_model.optimizer.set_weights(weight_values)
        initial_epoch = sc.config['continue_training_epoch']


logger.info("Model built")
# {eval_loss:.3f}
filepath= os.path.join(
    weights_path,
    "w-{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5"
    )


# In[ ]:




tensorflow_trace = sc.config["tensorflow_trace"]
if tensorflow_trace:
    tensorboard_profile_batch = 2
else:
    tensorboard_profile_batch = 0
verbose = sc.config["training_verbose"]

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, 
                             save_best_only=True, mode='min', 
                             save_weights_only=True)
tensorboard = TensorBoard(log_dir=log_path, 
                          histogram_freq=1,  
                          profile_batch = tensorboard_profile_batch,
                          write_graph=tensorflow_trace,
                          write_images=tensorflow_trace)

save_optimizer = model.resources.SaveOptimizerCallback(weights_path)
# evaluation = model.resources.AdditionalValidationSet(fp_dataset_eval, 
#                                                      "eval", 
#                                                      verbose = 0)

print_logs = LambdaCallback(
    on_epoch_end = lambda epoch, logs: print(logs)
    )

json_log = open(os.path.join(weights_path, 'loss_log.json'),
 mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)
#

callbacks_list = [
    #evaluation, 
                  tensorboard, 
                  print_logs, 
                  json_logging_callback,
                  checkpoint, 
                  save_optimizer]


# In[ ]:



logger.info("Training - start")
transcoder_model.fit(x=fp_dataset_train, 
          epochs=epochs, 
          #batch_size=sc.config['batch_size'],
          steps_per_epoch=training_steps,
          callbacks = callbacks_list,
          validation_data = fp_dataset_val,
          validation_steps = validation_steps,
          initial_epoch = initial_epoch,
          verbose = verbose)
logger.info("Training - done")
fp_db.close()

logger.info("training end")


# In[ ]:


fp_dataset_train, fp_dataset_val

