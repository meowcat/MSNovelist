# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:10:34 2021

@author: stravsm
"""

import os

from . import fingerprinting as fpr
from . import fingerprint_map as fpm


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import base64

import pickle

from . import fingerprinting as fpr
from . import fingerprint_map as fpm

from collections import Counter

from warnings import warn


import numpy as np
import pandas as pd
from random import random, seed
import logging
logger = logging.getLogger("MSNovelist")


import h5py


# Below is auxiliary code that can hopefully go away in the future
import smiles_config as sc
import os
from tqdm import tqdm
import dill
import pickle
import sqlite3
import re


# db_pubchem_path = os.path.join(
#     sc.config["base_folder"],
#     "pubchem-cid-smiles-inchikey.db")

try:
    if 'db_pubchem' in sc.config:
        db_pubchem_path = sc.config["db_pubchem"]
    else:
        db_pubchem_path = os.path.join(
            sc.config["base_folder"],
            "pubchem_ref/pubchem_ref.db")

    db_pubchem = sqlite3.connect(db_pubchem_path)
except:
    warn("PubChem database not found or not connected (read-only path?)")



def get_smiles_pubchem(df, db_pubchem, verbose=False):
    sql_str = '''
    SELECT smiles
    FROM inchi_to_smiles 
    WHERE inchikey1 = ?
    '''
    failures = Counter()
    def _get_smiles_pubchem(inchikey):
        sql_args = [inchikey[:14]]    
        try:
            with db_pubchem as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
                data = cursor.fetchall()
                if len(data) > 1:
                    if verbose:
                        warn(f"{inchikey}: More than one SMILES retrieved")
                    failures.update('m')
                if len(data) < 1:
                    if verbose:
                        warn(f"{inchikey}: No SMILES retrieved")
                    failures.update('f')
                    return ""
                return data[0][0]
        except Error as e:
            warn(e)
            return ""
    df["smiles_in"] = [_get_smiles_pubchem(i) for i in tqdm(df["inchikey"])]
    if len(failures) > 0:
        logger.info(f"Not in lookup database: {failures} entries")
    return df
  
                
import requests

def pubchem_standardize_request(inchi):
    return {
        'submitjob': 'submitjob',
        'structure': 'inchi',
        'structureinchi': inchi,
        'output': 'smiles'
    }
pubchem_standardize_url = "https://pubchem.ncbi.nlm.nih.gov/standardize/standardize.cgi"
def pubchem_standardize_(inchi):
    return requests.post(
        pubchem_standardize_url, 
        pubchem_standardize_request(inchi))

def pubchem_standardize(inchi):
    return bytes.decode(
        pubchem_standardize_(inchi).content).replace('\n','')


def standardize_missing_smiles_pubchem(df, inplace = True):
    '''
    
    For data that was not in the pubchem DB, find standardized SMILES
    via the webinterface.

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if not inplace:
        df = df.copy()
    smiles_missing = df["smiles_in"] == ""
    inchi_to_retrieve = df["inchi"][smiles_missing]
    df["smiles_in"][smiles_missing] = [pubchem_standardize(i) 
                                       for i in tqdm(inchi_to_retrieve)]
    if not inplace:
        return df
    

def get_formula(m, h = True):
    m_ = Chem.AddHs(m)
    c = Counter(atom.GetSymbol() for atom in m_.GetAtoms())
    if h == False:
        c['H'] = 0
    return c
    

def process_df(block,
               fingerprinter,
               construct_from = "inchi",
               write = ["smiles_generic",
                        "smiles_canonical",
                        "mol","inchikey", "inchikey1","mf"],
               block_id = None):
    if construct_from == "inchi":
        inchi = block["inchi"]
        mol = [Chem.MolFromInchi(m) for m in inchi]
        smi = [Chem.MolToSmiles(m) for m in mol]
        block["smiles_in"] = smi
    elif construct_from == "smiles":
        block["smiles_in"] = block["smiles"]
    else:
        raise ValueError("No valid input, specify smiles or inchi")
    # Run input smiles (from smiles or from inchi) through Java SIRIUS wrapper
    # to get canonical SMILES for all of them.
    data = fingerprinter.process(block["smiles_in"].fillna("").tolist(), 
                                 calc_fingerprint = False)

    block = block.iloc[[x["data_id"] for x in data],:].copy()
    smiles_generic = [d["smiles_generic"] for d in data]
    if "smiles_generic" in write:
        block["smiles_generic"] = smiles_generic
    smiles = [d["smiles_canonical"] for d in data]
    if "smiles_canonical" in write:
        block["smiles_canonical"] = smiles
    # build molecules
    mol = [Chem.MolFromSmiles(smiles_) for smiles_ in smiles]
    if "mol" in write:
        block["mol"] = mol
    if "inchikey" in write:
        block["inchikey"] = [
                Chem.MolToInchiKey(m) if m is not None
                else ""
                for m in mol]
        if "inchikey1" in write:
            block["inchikey1"] = [
                    ik.split('-')[0] for ik in block["inchikey"]
                    ]
    if "mf" in write:
        block["mf"] = [
                get_formula(m) if m is not None
                else []
                for m in mol]
    if block_id is not None:
        block["block_id"] = block_id
    return block

class FpDatabase:
     
    ext_mapping = {}
    
    def __init__(self, db_file, config = None):
        self.db_file = db_file
        self.config = config
    
    def insert_fp(self, fp_item):
        raise NotImplementedError("Abstract DB doesn't implement insert_fp")
        
    def insert_fp_multiple(self, fp_items):
        raise NotImplementedError("Abstract DB doesn't implement insert_fp_multiple")
        
    def close(self):
        pass
    
    def get_grp(self, grp):
        raise NotImplementedError("Abstract DB doesn't implement get_grp")

    def get_all(self):
        raise NotImplementedError("Abstract DB doesn't implement get_all")
    
    def delete_grp(self, grp):
        raise NotImplementedError("Abstract DB doesn't implement delete_grp")
    
    def set_grp(self, grp, index, fold = False):
        raise NotImplementedError("Abstract DB doesn't implement set_grp")
        
    def get_pipeline_options(self):
        options = {}
        if 'pipeline_options' in self.config:
            options.update(self.config['pipeline_options'])
        return options
        
    @classmethod
    def register_mapping(cls, filetype, constructor):
        cls.ext_mapping[filetype] = constructor
    
    @classmethod
    def load_from_config(cls, config):
        '''
        Load database with the appropriate driver (csv or sqlite).

        Parameters
        ----------
        cls : TYPE
            DESCRIPTION.
        config : TYPE
            DESCRIPTION.

        Returns
        -------
        db : TYPE
            DESCRIPTION.

        '''
        # Just a path: choose based on extension
        if isinstance(config, str):
            db_path = config
            db_config = {}
        elif isinstance(config, dict):
            db_path = config["path"]
            db_config = config
        _, ext = os.path.splitext(db_path)
        # ext_mapping = {
        #     ".csv": FpDatabaseCsv,
        #     ".db": FpDatabaseSqlite,
        #     ".pkl": fp_database_pickle,
        #     ".h5": FpDatabaseHdf5}
        db = cls.ext_mapping[ext](db_path, db_config)
        return db