# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 14:01:38 2020

@author: stravsm
"""


import os
import sys
sys.path.append(os.environ['MSNOVELIST_BASE'])

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import infrastructure.score as msc
import infrastructure.utils as utils

from fp_management import fingerprint_map as fpm
import smiles_config as sc

import time
import pickle
# Setup logger
import logging
import h5py
from tqdm import tqdm
import glob

from rdkit import Chem

from scipy import stats


logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("identity_ranking_with_metrics startup")

eval_id = str(int(time.time()))
eval_counter = 0
pickle_id = eval_id
if sc.config['eval_id'] != '':
    eval_id = sc.config['eval_id']
if sc.config['eval_counter'] != '':
    eval_counter = sc.config['eval_counter']
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
weights = sc.config['weights_folder'] + sc.config['weights']


sc.config.setdefault('cv_fold', 0)
cv_fold = sc.config["cv_fold"]
evaluation_set_ = sc.config['evaluation_set']
evaluation_set = f"fold{cv_fold}-{evaluation_set_}"

evaluation_logger = utils.EvaluationLogger("topn", sc.config, 
                                           eval_id, eval_counter, pickle_id)

pipeline_encoder = sc.config['pipeline_encoder']
pipeline_reference = sc.config['pipeline_reference']

picklepath = sc.config.get("evaluation_picklepath", {})
if sc.config['eval_counter'] != '':
    pickle_id = sc.config['eval_id'] + "-" + sc.config['eval_counter']
    picklelist = glob.glob(os.path.join(sc.config["eval_folder"], f"eval_{pickle_id}*.pkl"))
    picklepath = {os.path.basename(x): x for x in picklelist}
else:
    sc.config["model_tag"] = list(picklepath.keys())[0]
             
              
def check_dict(v):
    if isinstance(v, dict):
        return v[evaluation_set]
    else:
        return v


results_complete = {k: pickle.load(open(pp, 'rb')) for k, pp in picklepath.items()}
results_complete = {k: check_dict(v) for k, v in results_complete.items()}
results_complete = pd.concat([r[["nn", "mol", "mol_ref", "mf", "mf_ref",
                                 "fingerprint", "fingerprint_ref", "fingerprint_ref_true", 
                                 "inchikey1", "inchikey1_ref",
                                 "score", "smiles", "smiles_ref"]].assign(source = k)
                    for k, r in results_complete.items()])

n_total_ = len(set(results_complete["nn"]))
kk = sc.config["eval_kk"]
k = sc.config["eval_k"]
f1_cutoff = 0

results_ok = results_complete.loc[results_complete["fingerprint"].notna()].copy()

n_results_ok = len(results_ok)
logger.info(f"Scoring fingerprints for {n_results_ok} results with correct MF")
#results_ok = results_complete.loc[[]].copy()
fp_map = fpm.FingerprintMap(sc.config["fp_map"])
scores = msc.get_candidate_scores()
results_ok = msc.compute_candidate_scores(results_ok, fp_map, 
                                          additive_smoothing_n = n_total_,
                                          f1_cutoff = f1_cutoff)
# Add "native" scoring i.e. using the decoder output directly
results_ok["score_decoder"] = results_ok["score"]
scores.update({"score_decoder": None})

logger.info(f"Scoring fingerprints - done")

for score in scores.keys():
    results_ok["rank_" + score] = results_ok.groupby("nn")[score].rank(ascending=False, method="first")

n_rank_cols = ["nn"] + ["rank_" + score for score in scores.keys()]
results_match_ranks = results_ok.loc[
    results_ok["inchikey1"] == results_ok["inchikey1_ref"]][n_rank_cols]

results_top_rank = results_match_ranks.groupby("nn").aggregate(np.min)

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)


results_ranks_ecdf = {}
for score in scores.keys():
    results_ranks_ecdf.update({score: ecdf(results_top_rank["rank_" + score])})

from matplotlib import pyplot as plt
#fig, ax = plt.subplot()
for score, data, i in zip(results_ranks_ecdf.keys(), 
                          results_ranks_ecdf.values(), 
                          range(len(results_ranks_ecdf))):                          
    plt.plot(data[0], data[1], color='C' + str(i))
plt.xscale("log")
plt.xlabel("rank inchikey1 hit")
plt.ylabel("ECDF")
plt.legend(scores.keys())
# for axis in [ax.xaxis, ax.yaxis]:
#     axis.set_major_formatter(ScalarFormatter())


results_complete["inchikey1_match"] = (
    results_complete["inchikey1"] == results_complete["inchikey1_ref"]
    )
results_complete["mf_match"] = (
    results_complete["mf"] == results_complete["mf_ref"]
    )
results_complete["valid_smiles"] = (
    results_complete["inchikey1"] != ''
    )


results_complete_lib = results_complete[["nn", "mol_ref", "fingerprint_ref", "mf_ref"]].groupby("nn").first()
results_with_ranks = results_complete_lib.join(results_match_ranks.set_index("nn"))
rank_cols = ["rank_" + score for score in scores.keys()]
results_with_ranks[rank_cols]= results_with_ranks[rank_cols].fillna(value = kk + 1)
results_per_candidate = results_complete.groupby(["nn"])
results_summary = results_per_candidate[
    ["valid_smiles", "mf_match", "inchikey1_match"]
    ].aggregate(np.sum)
results_summary_bool = results_summary >= 1
results_summary_bool.mean()


    
    # Add a table for valid and correct-MF SMILES per challenge


results_mf_summary = results_summary.copy()
results_mf_summary.reset_index(inplace=True)
results_mf_summary.set_index("nn", inplace=True)
all_res = results_complete.groupby("nn").first()[
    ["fingerprint", "fingerprint_ref", "fingerprint_ref_true",
     "mol_ref", "mf_ref"]]
results_mf_summary = results_mf_summary.join(all_res)
results_mf_summary = results_mf_summary.loc[results_mf_summary["fingerprint_ref_true"].notna()]

results_mf_summary = msc.compute_fp_quality_mw(results_mf_summary, fp_map)

results_mf_summary.reset_index(inplace=True)

results_keys = ["valid_smiles", "mf_match"]
for key in results_keys:
    results_mf_summary["rank"] = results_mf_summary[key].rank(
        ascending=False, method='first')
    results_mf_summary["value"] = results_mf_summary[key] / kk
    results_mf_summary["eval_score"] = key
    results_mf_summary["eval_metric"] = "rank"
        
    evaluation_logger.append_csv(
        "valid_mf",
        results_mf_summary[["nn", "rank", "value", "eval_score",
                            "eval_metric", "predicted_fp_quality", "mol_weight"]]
        )


results_topk_ = {}
results_top_rank = results_with_ranks[rank_cols]#.aggregate(np.min)
for rank in [1, 1.99, 5, 10, 20]:
    results_topk_.update({'top_' + str(rank): (results_top_rank <= rank).mean()})
results_topk = pd.DataFrame(results_topk_)

#########################
# score_fivenum and tanimoto_fivenum indicators:
# score_fivenum: five-number statistic of best scores reached per query
# tanimoto_fivenum: five-number statistics of tanimoto similarity of 
#    top-scoring candidate per query,
#    i.e. compound similarity of best match
#########################
score_fivenum_ = {}
tanimoto_fivenum_ = {}

for score in scores.keys():
    # get rank-1 candidates for each query
    results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
    results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]

    # get score_fivenum for rank-1 scandidates
    score_fivenum_.update({"fivenum_" + score: results_top1_score[score].describe()})
    # get fingerprints and calculate tanimoto for rank-1 candidates
    fingerprint_candidate_rank1 = np.concatenate(results_top1_score["fingerprint"].tolist())
    fingerprint_truematch = np.concatenate(results_top1_score["fingerprint_ref_true"].tolist())
    results_top1_score["match_tanimoto"] = list(map(
        lambda x: msc.fp_tanimoto(x[0], x[1]), zip(fingerprint_candidate_rank1, fingerprint_truematch)
        ))
    tanimoto_fivenum_.update({"tanimoto_" + score: results_top1_score["match_tanimoto"].describe()})
    
score_fivenum = pd.DataFrame.from_records(score_fivenum_).transpose()
tanimoto_fivenum = pd.DataFrame.from_records(tanimoto_fivenum_).transpose()
print(score_fivenum)
print(tanimoto_fivenum)

    


#########################
# Top-n evaluation: % spectra < rank n € {1, 1.99, 5, 10, 20}
# for different scores.
#########################

logger.info("Evaluation overall:")
print(results_summary_bool.mean())
logger.info("Evaluation top-n ECDF:")
print(results_topk)

evaluation_logger.append_txt(
    key = 'summary',
    data = {'Evaluation set': evaluation_set,
     'Beam width': k,
     'Top-k': kk,
     'Pipeline': pipeline_encoder,
     'n': n_total_,
     'Total MF OK': n_results_ok,
     'Summary SMILES': results_summary_bool.mean(),
     'Results top-k ECDF': results_topk,
     'Best scores summary': score_fivenum,
     'Rank-1 tanimoto summary': tanimoto_fivenum
     })


evaluation_logger.append_csv("identity", pd.DataFrame(results_summary_bool.mean()).transpose())
evaluation_logger.append_csv("score_fivenum", score_fivenum)
evaluation_logger.append_csv("tanimoto_fivenum", tanimoto_fivenum)
evaluation_logger.append_csv("ranks", results_topk)


if sc.config["eval_detail"]:
    
    for score in scores.keys():
        # get rank-1 candidates for each query
        results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
        results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]
        results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)
        # get score_fivenum for rank-1 scandidates
        results_top1_score["rank"] = results_top1_score[score].rank(ascending=False, method='first')
        results_top1_score["value"] = results_top1_score[score]
        results_top1_score["eval_score"] = score
        results_top1_score["eval_metric"] = "score"
        evaluation_logger.append_csv(
            "score_quantiles",
            results_top1_score[["nn", "rank", "value", "eval_score", 
                                "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )
    
    for score in scores.keys():
        # get rank-1 candidates for each query
        results_top1_score = results_ok.loc[results_ok["rank_" + score] == 1].copy()
        results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]
        results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)
        # get score_fivenum for rank-1 scandidates
        results_top1_score["rank"] = results_top1_score["score_mod_platt"].rank(ascending=False, method='first')
        results_top1_score["value"] = results_top1_score["score_mod_platt"]
        results_top1_score["eval_score"] = score
        results_top1_score["eval_metric"] = "score"
        evaluation_logger.append_csv(
            "platt_quantiles",
            results_top1_score[["nn", "rank", "value", "eval_score", 
                                "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )
        
    
    
    for score in scores.keys():
        # get rank-1 candidates for each query
        results_ref_mols = results_per_candidate.first()[["mol_ref", "fingerprint_ref_true", "fingerprint_ref"]]
        results_top1_score = results_top_rank.copy()
        results_top1_score = results_top1_score.join(results_ref_mols)
        results_top1_score = results_top1_score.loc[results_top1_score["fingerprint_ref_true"].notna()]

        results_top1_score.reset_index(inplace=True)
        results_top1_score = msc.compute_fp_quality_mw(results_top1_score, fp_map)

        # get score_fivenum for rank-1 scandidates
        results_top1_score["rank"] = results_top1_score["rank_" + score].rank(ascending=True, method='first')
        results_top1_score["value"] = results_top1_score["rank_" + score]
        results_top1_score["eval_score"] = score
        results_top1_score["eval_metric"] = "rank"
        evaluation_logger.append_csv(
            "rank_quantiles",
            results_top1_score[["nn", "rank", "value", "eval_score", 
                                "eval_metric", "predicted_fp_quality", "mol_weight"]],
            )


    # Spearman rank corr
    

    def weighted_spearman_loss(yhat, y):
        spearman_loss = np.square(yhat - y)
        weight = 1/y
        return np.sum(weight * spearman_loss)
        
    results_ref_mols = results_per_candidate.first()[["mol_ref", "fingerprint_ref_true", "fingerprint_ref"]]
    results_ref_mols = results_ref_mols.loc[results_ref_mols["fingerprint_ref_true"].notna()]

    results_cor_group = results_ok.groupby("nn")[["rank_score_decoder","rank_score_mod_platt"]]
    results_cor = results_cor_group.corr(weighted_spearman_loss).unstack().iloc[:,1:2]
    results_cor.columns = ["value"]
    results_cor["eval_score"] = "weighted_spearman_loss"
    results_cor = results_cor.join(results_ref_mols)
    #results_cor = results_cor.loc[results_cor["fingerprint"].notna()]
    results_cor = results_cor.loc[results_cor["fingerprint_ref"].notna()]
    results_cor = results_cor.loc[results_cor["fingerprint_ref_true"].notna()]
    results_cor = results_cor.copy()
    results_cor = msc.compute_fp_quality_mw(results_cor, fp_map)

    evaluation_logger.append_csv(
        "rank_corr",
        results_cor
        )

    results_cor_group = results_ok.groupby("nn")[["rank_score_decoder","rank_score_mod_platt"]]
    results_cor = results_cor_group.corr('spearman').unstack().iloc[:,1:2]
    results_cor.columns = ["value"]
    results_cor["eval_score"] = "spearman_cor"
    results_cor = results_cor.join(results_ref_mols)
    #results_cor = results_cor.loc[results_cor["fingerprint"].notna()]
    results_cor = results_cor.loc[results_cor["fingerprint_ref"].notna()]
    results_cor = results_cor.loc[results_cor["fingerprint_ref_true"].notna()]
    results_cor = results_cor.copy()
    results_cor = msc.compute_fp_quality_mw(results_cor, fp_map)

    evaluation_logger.append_csv(
        "rank_corr",
        results_cor
        )
    
    results_ok_out = results_ok.copy()
    results_ok_out.drop(["fingerprint","fingerprint_ref","fingerprint_ref_true","mol", "mol_ref"], axis = 1, inplace = True)
    results_ok_out["mf"] = results_ok_out.mf.apply(msc.formula_to_string)
    results_ok_out["mf_ref"] = results_ok_out.mf_ref.apply(msc.formula_to_string)
    evaluation_logger.append_csv("results_ok_ranked", results_ok_out)

