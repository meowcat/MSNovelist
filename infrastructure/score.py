# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 21:20:42 2018

@author: dvizard
"""


import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from IPython.display import Image


from rdkit.Avalon import pyAvalonTools

from rdkit import DataStructs
from rdkit.Chem import rdFMCS

from collections import Counter

#import smiles_config as sc
import operator
from tqdm import tqdm

def fp_tanimoto(fp1, fp2):
    if len(fp1) != len(fp2):
        return 0
    fp1_ = to_bitvect(fp1)
    fp2_ = to_bitvect(fp2)
    tanimoto = DataStructs.FingerprintSimilarity(fp1_, fp2_)
    return tanimoto

def to_bitvect(fp):
    bv = DataStructs.ExplicitBitVect(len(fp))
    [bv.SetBit(i) for i in range(len(fp)) if fp[i]]
    return bv

def rd_fingerprints(m):
    return (np.array(AllChem.RDKFingerprint(m)),
            np.array(pyAvalonTools.GetAvalonFP(m)),
            np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2))
            )
    

def score_platt(predicted, candidate, stats=None, f1_cutoff = 0.5):
    '''
    

    Parameters
    ----------
    predicted : np.array(m) dtype="float"
        1-D array of a probabilistic fingerprints of m bits.
    candidate : np.array(n,m)
        Array of n candidate deterministic (binary) fingerprints of m bits.

    Returns
    -------
    Platt score P  = Product_{i in 1..m}(predicted_i if candidate_i == 1,
                                    1-predicted_i if candidate_i == 0)
    Note that the score is zero if any predicted value is 0 or 1 and 
    mismatches with the true bit (which makes sense if the probability truly
    is zero, but it means never use this with non-probabilistic fingerprints)

    '''
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted
    
    sc = predicted * candidate + (1-predicted) * (1-candidate)
    sc = sc[:, stats[:,2] > f1_cutoff]
    sc = np.log(sc)
    return np.sum(sc, axis=1)
    

def score_unit(predicted, candidate, stats=None, f1_cutoff = 0.5):
    '''
    Unit score, simply counts the number of matching features between
    predicted fingerprint and candidates. For non-probabilistic FPs, this
    is equivalent to Platt score.

    Parameters
    ----------
    predicted : np.array(m) dtype="float"
        1-D array of a probabilistic fingerprints of m bits.
    candidate : np.array(n,m)
        Array of n candidate deterministic (binary) fingerprints of m bits.
    stats : TYPE, optional
        Not needed here, but can be supplied for signature equivalence to
        other scoring functions.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted
    
    sc = (np.round(predicted_) * candidate +
          np.round(1-predicted_) * (1-candidate))
    sc = sc[:, stats[:,2] > f1_cutoff]
    
    return np.sum(sc, axis=1)
    
def score_unit_pos(predicted, candidate, stats=None, f1_cutoff = 0.5):
    '''
    Unit_pos score, simply counts the number of matching "1" features between
    predicted fingerprint and candidates. For non-probabilistic FPs, this
    is equivalent to Platt score.

    Parameters
    ----------
    predicted : np.array(m) dtype="float"
        1-D array of a probabilistic fingerprints of m bits.
    candidate : np.array(n,m)
        Array of n candidate deterministic (binary) fingerprints of m bits.
    stats : TYPE, optional
        Not needed here, but can be supplied for signature equivalence to
        other scoring functions.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted
    
    sc = (np.round(predicted_) * candidate)
    sc = sc[:, stats[:,2] > f1_cutoff]
    
    return np.sum(sc, axis=1)
    

def score_mod_platt(predicted, candidate, stats, f1_cutoff = 0.5):
    '''
    "Modified Platt" score as in Dührkop et al. 10.1073/pnas.1509788112
    

    Parameters
    ----------
    predicted : np.array(m) dtype="float"
        1-D array of a probabilistic fingerprints of m bits.
    candidate : np.array(n,m)
        Array of n candidate deterministic (binary) fingerprints of m bits.
    stats : np.array(m, 2)
        An array containing columns (sensitivity, specificity) for each 
        fingerprint bit.
    Returns
    -------
    We'll see

    '''
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted

    predicted_r = np.round(predicted_)
    b11 = predicted_r * candidate           # 1 if p_i >0.5, c_i = 1
    b10 = predicted_r * (1-candidate)       # 1 if p_i >0.5, c_i = 0
    b01 = (1-predicted_r) * candidate       # 1 if p_i <0.5, c_i = 1
    b00 = (1-predicted_r) * (1-candidate)   # 1 if p_i <0.5, c_i = 0
    
    score_a = b11 * ( 0.75 * np.log(predicted) + 0.25 * np.log(1-stats[:,0]))
    score_b = b10 * ( 0.75 * np.log(1-predicted))
    score_c = b01 * ( 0.75 * np.log(predicted))
    score_d = b00 * ( 0.75 * np.log(1-predicted) + 0.25 * np.log(1-stats[:,1]))
              
    sc = score_a + score_b + score_c + score_d
    sc = sc[:, stats[:,2] > f1_cutoff]
    return np.sum(sc, axis=1)

def score_rel_mod_platt(predicted, candidate, stats, f1_cutoff = 0.5):
    '''
    Scales mod_platt between not precisely, but roughly 0,1

    '''
    
    # shift the mod_platt to the almost-best achievable mod_platt
    score_high_mod_platt = score_mod_platt(predicted, predicted.round(),
                                            stats, f1_cutoff)
    score_low_mod_platt = score_mod_platt(predicted, 1-predicted.round(),
                                            stats, f1_cutoff)
    score_true_mod_platt = score_mod_platt(predicted, candidate, 
                                           stats, f1_cutoff)
    
    return ((score_true_mod_platt - score_low_mod_platt) / 
            (score_high_mod_platt - score_low_mod_platt))

def score_lim_mod_platt(predicted, candidate, stats, f1_cutoff = 0.5):
    '''
    Scales mod_platt to a maximum of not precisely, but roughly 0

    '''
    
    # shift the mod_platt to the almost-best achievable mod_platt
    score_high_mod_platt = score_mod_platt(predicted, predicted.round(),
                                            stats, f1_cutoff)
    score_true_mod_platt = score_mod_platt(predicted, candidate, 
                                           stats, f1_cutoff)
    
    return score_true_mod_platt - score_high_mod_platt
    
 
def score_max_likelihood(predicted, candidate, stats, f1_cutoff = 0.5):
    '''
    Maximum likelihood score as in Dührkop et al. 10.1073/pnas.1509788112
    
    This one doesn't need the probabilistic fingerprint (but works the same).

    Parameters
    ----------
    predicted : np.array(m) dtype="float"
        1-D array of a probabilistic fingerprints of m bits.
    candidate : np.array(n,m)
        Array of n candidate deterministic (binary) fingerprints of m bits.
    stats : np.array(m, 2)
        An array containing columns (sensitivity, specificity) for each 
        fingerprint bit.
    Returns
    -------
    We'll see

    '''
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted
    predicted_r = np.round(predicted_)
    b11 = predicted_r * candidate           # 1 if p_i >0.5, c_i = 1
    b10 = predicted_r * (1-candidate)       # 1 if p_i >0.5, c_i = 0
    b01 = (1-predicted_r) * candidate       # 1 if p_i <0.5, c_i = 1
    b00 = (1-predicted_r) * (1-candidate)   # 1 if p_i <0.5, c_i = 0
    
    score_a = b11 * ( np.log(stats[:,0]))
    score_b = b10 * ( np.log(1-stats[:,1]))
    score_c = b01 * ( np.log(1-stats[:,0]))
    score_d = b00 * ( np.log(stats[:,1]) ) 
    
    
    sc = score_a + score_b + score_c + score_d
    sc = sc[:, stats[:,2] > f1_cutoff]
    return np.sum(sc, axis=1)
    
def score_tanimoto(predicted, candidate, stats = None, f1_cutoff = 0.5):
    
    if predicted.ndim < 2:
        predicted_ = np.expand_dims(predicted, 0)
    else:
        predicted_ = predicted
    predicted_r = np.round(predicted_)
    
    b11 = predicted_r * candidate           # 1 if p_i >0.5, c_i = 1
    b10 = predicted_r * (1-candidate)       # 1 if p_i >0.5, c_i = 0
    b01 = (1-predicted_r) * candidate       # 1 if p_i <0.5, c_i = 1
    b00 = (1-predicted_r) * (1-candidate)   # 1 if p_i <0.5, c_i = 0
    
    numerator = np.sum(b11[:, stats[:, 2] > f1_cutoff], axis=1)
    denominator = np.sum((b10 + b01 + b11)[:, stats[:, 2] > f1_cutoff], axis=1)
    
    return numerator / denominator



_candidate_scores = {
    "score_mod_platt": score_mod_platt,
    "score_unit": score_unit,
    "score_unit_pos": score_unit_pos,
    "score_platt": score_platt,
    "score_max_likelihood": score_max_likelihood,
    "score_tanimoto": score_tanimoto,
    "score_rel_mod_platt": score_rel_mod_platt,
    "score_lim_mod_platt": score_lim_mod_platt
    }

def get_candidate_scores(): 
    return _candidate_scores.copy()

def compute_candidate_scores(df, fp_map, scores = None, 
                             additive_smoothing_n = None,
                             compute_similarity = False,
                             f1_cutoff = 0.5):
    '''
    Wrapper for scoring, adds columns with all calculated scores to the input dataframe
    which must contain fingerprint_ref, fingerprint, and fingerprint_ref_true columns
    fingerprint_ref: the query fingerprint
    REMOVED: fingerprint_ref_true: the fingerprint of the true match (note: isn't actually used...)
    fingerprint: the fingerprint of the candidate

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    fp_map : TYPE
        DESCRIPTION.
    scores : TYPE, optional
        DESCRIPTION. The default is candidate_scores.
    additive_smoothing_n : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    df = df.copy()
    if scores is None:
        scores = get_candidate_scores()
    
    # Treat case with no items, which would otherwise fail because
    # there are no arrays to stack/concatenate
    
    if len(df) == 0:
        for score in scores.keys():
            df[score] = []
        return df
        
    fingerprint_ref = np.stack(df["fingerprint_ref"])
    fingerprint_candidate = np.concatenate(df["fingerprint"].tolist())
    #fingerprint_true = np.concatenate(df["fingerprint_ref_true"].tolist())
    fingerprint_candidate_map = fingerprint_candidate[:,fp_map.positions]
    #fingerprint_true_map = fingerprint_true[:,fp_map.positions]
    # Additive smoothing
    if additive_smoothing_n is not None:
        alpha = 1/additive_smoothing_n
        fingerprint_ref = (fingerprint_ref + alpha) / (1 + 2*alpha)

    for score, score_fun in scores.items():
        df[score] = score_fun(fingerprint_ref, 
                                      fingerprint_candidate_map,
                                      fp_map.stats,
                                      f1_cutoff)
    
    return df

def compute_fp_quality_mw(df, fp_map):
    def _compute_fp_quality(row): 
        return fp_tanimoto(
            row["fingerprint_ref_true"].reshape((-1))[fp_map.positions], 
            row["fingerprint_ref"].reshape((-1)).round())
    df["predicted_fp_quality"] = df.apply(_compute_fp_quality, axis=1)
    df["mol_weight"] = df["mol_ref"].apply(lambda x: Chem.rdMolDescriptors.CalcExactMolWt(x))
    return df



def get_mcs(row):
    mol = row["mol"]
    mol_ref = row["mol_ref"]
    mcs = rdFMCS.FindMCS(
        [mol, mol_ref], 
        matchValences=False, 
        ringMatchesRingOnly=True, 
        completeRingsOnly=False, 
        verbose=True,
        timeout=10)
    return mcs
    

def score_mcs(row, 
              atom_factor = 1, 
              bond_factor = 1):
    mol = row["mol"]
    mol_ref = row["mol_ref"]
    mcs = row["mcs"]
    # sum_mol = (atom_factor * mol.GetNumAtoms() + 
    #            bond_factor * mol_ref.GetNumBonds)
    sum_ref = (atom_factor * mol.GetNumAtoms() + 
               bond_factor * mol_ref.GetNumBonds())
    sum_mcs = (atom_factor * mcs.numAtoms +
               bond_factor * mcs.numBonds)
    score = sum_mcs / sum_ref
    return score


def score_fp(row):
    return fp_tanimoto(row["fingerprint"].reshape((-1)), 
                           row["fingerprint_ref_true"].reshape((-1)))


def compute_similarity(df, fp_map, compute_mcs = False, 
                       mcs_atom_factor = 1, mcs_bond_factor = 1):
    if compute_mcs:
        df["mcs"] = df.apply(get_mcs, axis=1)
        df["similarity_mcs"] = df.apply(
            lambda row: score_mcs(row, mcs_atom_factor, mcs_bond_factor), 
            axis=1)
    df["similarity_fp"] = df.apply(
        lambda row: score_fp(row), 
        axis=1)
    return df


def formula_to_string(mf):
    return ''.join([k + str(v) for k, v in mf.items()])


