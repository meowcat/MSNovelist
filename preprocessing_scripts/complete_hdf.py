# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:47:22 2021

@author: stravsm
"""

from rdkit import Chem
from tqdm import tqdm
import h5py

## Use:
# complete_hdf(filename)
# fills inchikeys, aromatic smiles, and molecular formulas in a h5 block




from fp_management import database as db
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm

import smiles_process as sp
import importlib
from importlib import reload
import smiles_config as sc
import numpy as np

# Setup logger
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("MSNovelist")
logger.setLevel(logging.INFO)
logger.info("HDF5 filler startup")

logger.info("initializing fingerprinter")
fp_map = fpm.FingerprintMap(sc.config["fp_map"])
fpr.Fingerprinter.init_instance(sc.config['fingerprinter_path'],
                                  sc.config['fingerprinter_threads'],
                                  capture = True)
fingerprinter = fpr.Fingerprinter.get_instance()

def complete_hdf(file):
    logger.info(f"processing file: {file}")
    d = h5py.File(file, mode='r+')

    logger.info("Building molecules")
    mol = list(map(Chem.MolFromSmiles, tqdm(d["smiles"])))
    
    
    if "inchikey1" not in d:
        logger.info("Building and adding InChIKeys")
        inchikey = map(Chem.MolToInchiKey, mol)
        inchikey1 = map(lambda x: x[:14], tqdm(inchikey, total = len(d["smiles"])))
        inchikey1_run = list(inchikey1)
        inchikey1_string = np.array(inchikey1_run, h5py.string_dtype())
        inchikey1_target = d.create_dataset("inchikey1", (len(inchikey1_string),), dtype= h5py.string_dtype())
        inchikey1_target[...] = inchikey1_string
    else:
        logger.info("InChIKeys already present - skipping")
    
    if "inchikey" not in d:
        d["inchikey"] = d["inchikey1"] # makes a link, if I get this correctly
    
    
    if "mf_text" not in d:
        logger.info("Building formulas")
        mf = list(map(Chem.rdMolDescriptors.CalcMolFormula, tqdm(mol)))
        mf_target = d.create_dataset("mf_text", (len(mf),), dtype= h5py.string_dtype())
        mf_target[...] = mf
    else:
        logger.info("Formulas already present - skipping")
    
    if "smiles_generic" not in d:
        logger.info("Generating aromatic SMILES")
        processed_smiles = fingerprinter.process(d["smiles"], calc_fingerprint = False)
        smiles_target = d.create_dataset("smiles_generic", (len(processed_smiles),), dtype= h5py.string_dtype())
        smiles_target[...] = list(map(lambda x: x['smiles_generic'], processed_smiles))
        if "smiles_canonical" not in d:
            smiles_target = d.create_dataset("smiles_canonical", (len(processed_smiles),), dtype= h5py.string_dtype())
            smiles_target[...] = list(map(lambda x: x['smiles_canonical'], processed_smiles))
    else:
        logger.info("SMILES already present - skipping")

    d.close()
    # inkey_ref = set(d2["inchikey1"][()])
    # inkey_ds = set(d["inchikey1"][()])
    # inkey_intersect = inkey_ref & inkey_ds



