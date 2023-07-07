# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:58:32 2020

@author: stravsm
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:36:30 2020

@author: stravsm
"""


import infrastructure.generator as gen
from collections import Counter
import numpy as np
import os

import importlib
from importlib import reload
from tqdm import tqdm
import os

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import tempfile

from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
import smiles_config as sc
import infrastructure.generator as gen
import infrastructure.decoder as dec

from pathlib import Path

# Randomness is relevant in the (rare) case of using stochastic sampling
random_seed = sc.config['random_seed_global']
if random_seed != '':
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.experimental.set_seed(random_seed)

project_path = sc.config['sirius_project_input']
queries = [f for f in os.scandir(project_path) if f.is_dir()]
queries = [f for f in queries if "fingerprints" in os.listdir(f)]
#queries_path  = [os.path.join(f.path, "fingerprints") for f in queries]


import time
from datetime import datetime
import pickle


from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import infrastructure.score as msc
import gc
import molmass
import shutil



# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("predict startup")

tf.get_logger().setLevel('ERROR')


eval_id = str(int(time.time()))
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
weights = sc.config['weights_folder'] + sc.config['weights']
picklepath = sc.config["eval_folder"] + "decode_" + pickle_id + ".pkl"
csv_path = sc.config["eval_folder"] + "decode_" + pickle_id + ".csv"
filelog_path = sc.config["eval_folder"] + "filelog_" + pickle_id

k = sc.config["eval_k"]
kk = sc.config["eval_kk"]
steps = sc.config["eval_steps"]


TRAINING_SET = sc.config['training_set']
VALIDATION_SET = sc.config['validation_set']
pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']

decoder_name = sc.config["decoder_name"]

# Should signal files be written to log progress?
output_filelog = "filelog" in sc.config
if output_filelog:
    os.mkdir(filelog_path)

# # Load dataset
# fp_db  = db.FpDatabase(sc.config['db_path'])
# fp_val = fp_db.get_grp(VALIDATION_SET)
fp_map = fpm.FingerprintMap(sc.config["fp_map"])
# fp_val = fp_val[:n_total_]

fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                fp_map,
                                sc.config['fingerprinter_threads'],
                                capture = False,
                                cache = sc.config['fingerprinter_cache'])
fingerprinter = fpr.Fingerprinter.get_instance()


# File for CSI:FingerID validation data
# We need to load some DB to get blueprints!
data_eval_ = sc.config["db_path_template"]
# Load mapping table for the CSI:FingerID predictors
# Load dataset and process appropriately
db_eval = db.FpDatabase.load_from_config(data_eval_)
dataset_val = db_eval.get_all()

pipeline_options =  db_eval.get_pipeline_options()
pipeline_options['fingerprint_selected'] = "fingerprint"

# Load dataset and sampler, apply sampler to dataset
# (so we can also evaluate from fingerprint_sampled)
fp_dataset_val_ = gen.smiles_pipeline(dataset_val, 
                                    batch_size = 1,
                                    map_fingerprints=False,
                                    **pipeline_options)


sampler_name = sc.config['sampler_name']
round_fingerprints = True
if sampler_name != '':
    logger.info(f"Sampler {sampler_name} settings loading")
    spl = importlib.import_module(sampler_name, 'fp_sampling')
    sf = spl.SamplerFactory(sc.config)
    round_fingerprints = sf.round_fingerprint_inference()
    logger.info(f"Sampler {sampler_name} settings loaded")

pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']
fp_dataset_val = gen.dataset_zip(fp_dataset_val_, 
                                 pipeline_encoder, pipeline_reference,
                                 **pipeline_options)
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

results = []

# Get an overlay fingerprint
if 'fp_overlay' in sc.config:
    fp_overlay_raw = fingerprinter.process([sc.config["fp_overlay"]], return_b64=True)
    fp_overlay = fpr.get_fp(fp_overlay_raw[0]['fingerprint'])
    fp_overlay_map = fp_overlay[0,fp_map.positions]
else:
    fp_overlay_map = np.zeros_like(fp_map.positions).reshape((1,-1))

m = len(queries)
logger.info(f"Processing {m} queries")
for i, query in enumerate(tqdm(queries)):
    query_path = os.path.join(query.path, "fingerprints")
    temp_fingerprints = tempfile.mkdtemp()
    shutil.unpack_archive(query_path, temp_fingerprints, "zip")
    fingerprints_path = os.listdir(temp_fingerprints)
    query_name = query.name
    mf = [path.split("_")[0] for path in fingerprints_path]
    fp_path = [os.path.join(temp_fingerprints, path) for path in fingerprints_path]
    fp = np.stack([np.genfromtxt(open(path, "r")) for path in fp_path])
    # Overlay the fingerprint with the proposed substructure of the user
    clip_max = np.max(fp)
    clip_min = np.min(fp)
    fp_orig = fp
    if sc.config['fp_overlay_method'] == 'add':
        # Add: simply add the overlay bits to the predicted fingerprint.
        # Should be used with a small substructure.
        fp = np.clip(fp_orig + fp_overlay_map, clip_min, clip_max)
    elif sc.config['fp_overlay_method'] == 'prob_merge':
        # Merge by probability: 
        # Use only confident predictions.
        # The closer a prediction is to 0.5, the more the overlay FP is used.
        # This needs a *complete* overlay FP, not just a small substructure!
        fp_orig_confidence = 2*np.abs(fp_orig - 0.5)
        fp = (1-fp_orig_confidence * fp_overlay_map) + (fp_orig_confidence * fp_orig)
        fp = np.clip(fp, clip_min, clip_max)
    elif sc.config['fp_overlay_method'] == "prob_add":
        # Add by probability: 
        # Add overlay 1 (but never subtract overlay 0) where the prediction
        # is low-confidence. Keep prediction 0 when it is high-confidence.
        fp = (fp_orig * fp_overlay_map) + fp_orig
        fp = np.clip(fp, clip_min, clip_max)



    fo = [Counter({e[0]: e[1] for e in molmass.Formula(mf_).composition()}) for mf_ in mf]
    fo_ = gen.mf_pipeline(fo).astype('float32')
    nh = fo_[:,-1]
    
    data = {'fingerprint_selected': fp, 
            'mol_form': fo_,
            'n_hydrogen': nh}
    n = len(fingerprints_path)
    
    # Initialize decoder
    decoder = dec.get_decoder(decoder_name)(
        model_encode, model_decode, steps, n, k, kk, config = sc.config)
        
    data_k = {key: tf.repeat(x, k, axis=0) for key, x in data.items()}
    states_init = model_encode.predict(data_k)
    # predict k sequences for each query.
    sequences, y, scores = decoder.decode_beam(states_init)
    seq, score, length = decoder.beam_traceback(sequences, y, scores)
    smiles = decoder.sequence_ytoc(seq)
    
    results_df = decoder.format_results(smiles, score)
    
    results_df = db.process_df(results_df, fingerprinter, construct_from = "smiles")
    
    results_mf_ref = pd.DataFrame({'n': range(n), 
                                   'mf': fo, 
                                   'fingerprint_ref': [fp] * n,
                                  'query': [query_name] * n 
                                  })
    results_df = results_df.join(results_mf_ref, on="n", rsuffix="_ref")
        # Keep only correct formula
    results_ok = results_df.loc[results_df["inchikey1"].notna()].loc[results_df["mf"] == results_df["mf_ref"]]
    results_ok["m"] = i
    results.append(results_ok)

    if output_filelog:
        (Path(filelog_path) / f'predict_{i}').write_text("")
 
logger.info(f"Processing {m} queries - fingerprinting results")
results_processed_ = []
for i, result in enumerate(tqdm(results)):
    result_processed = fingerprinter.process_df(result)
    results_processed_.append(result_processed)
    if output_filelog:
        (Path(filelog_path) / f'fingerprint_{i}').write_text("")

logger.info(f"Processing {m} queries - merging")
results_processed = pd.concat(results_processed_)
#pickle.dump(results_processed, open(picklepath, "wb"))


#results_complete = pd.concat(results_processed)

logger.info(f"Processing {m} queries - computing scores")
results_copy = results_processed.copy()
results_copy["fingerprint_ref"] = results_copy["fingerprint_ref"].apply(lambda x: x[0].astype("float32"))

del(results_processed)
del(results)
del(results_processed_)

results_scores = msc.compute_candidate_scores(results_copy, fp_map, additive_smoothing_n = 5000)
results_scores["score_decoder"] = results_scores["score"]

if output_filelog:
    (Path(filelog_path) / f'scores').write_text("")



scores = msc.get_candidate_scores()
scores["score_decoder"] = 0

for score in scores.keys():
    results_scores["rank_" + score] = results_scores.groupby(["m", "n"])[score].rank(ascending=False, method='first')


logger.info(f"Processing {m} queries - exporting")
pickle.dump(results_scores, open(picklepath, "wb"))



export_columns = ["m", "query", "n", "mf_text", "k", "id", "mz", 
                  "score_decoder", "score_lim_mod_platt", "score_mod_platt",
                   "rank_score_decoder", "rank_score_lim_mod_platt",
                   "inchikey1", "smiles"]

from rdkit.Chem import rdMolDescriptors
results_export = results_scores.copy()
results_export["mf_text"] = [rdMolDescriptors.CalcMolFormula(m) for m in tqdm(results_export["mol"])]
results_export["mz"] = [rdMolDescriptors.CalcExactMolWt(m) + 1.0072 for m in tqdm(results_export["mol"])]
results_export = results_export[export_columns]
results_export.to_csv(csv_path, index=False)
