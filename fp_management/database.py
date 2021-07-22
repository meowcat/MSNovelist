# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:01:11 2019

@author: dvizard
"""

import sqlite3
from sqlite3 import Error

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

import re
import h5py


# Below is auxiliary code that can hopefully go away in the future
import smiles_config as sc
import os
from tqdm import tqdm
import dill
import pickle



    
    

class FpItem:
    '''
    TODO: I believe this is obsolete and not actuall used anymore for anything.
    Make sure and then delete.
    '''
    
    def __init__(self, inchikey1, inchikey = None,
                 smiles_generic = None, smiles_canonical = None,
                 fp = None, mol = None, mf = None, mf_text = None,
                 source='', grp=''):
        self.inchikey1 = inchikey1
        self.inchikey = inchikey
        self.smiles_generic = smiles_generic
        self.smiles_canonical = smiles_canonical
        self.fp = fp
        self.fp_degraded = None
        self.mol = mol
        self.mf = mf
        self.mf_text = mf_text
        self.source = source
        self.grp = grp
       
    @classmethod
    def fromSiriusFp(cls, smiles_generic,
                     smiles_canonical, fp, source = '', grp = '', b64 = True):
        mol = Chem.MolFromSmiles(smiles_canonical)
        if mol is not None:
            inkey = Chem.MolToInchiKey(mol)
            inkey1 = inkey.split('-')[0]
            mf = get_formula(mol)
            mf_text = rdMolDescriptors.CalcMolFormula(mol)
        else:
            inkey = ''
            inkey1 = ''
            mf = Counter()
        if b64:
            fp = base64.b64decode(fp)
        return FpItem(inkey1, inkey, smiles_generic,
                      smiles_canonical, fp, mol, mf, mf_text, source, grp)
    
    @classmethod
    def fromDbRow(cls, row):
        [inkey1, inkey, smiles_generic, smiles_canonical, fp, source, grp] = [
                row[x] for x in ["inchikey1", "inchikey", "smiles_generic",
                                 "smiles_canonical", "fingerprint", "source", "grp"]]
        return(FpItem(inkey1, inkey, smiles_generic, smiles_canonical,
                      fp, source, grp))


from .fp_database import *
from .fp_database_hdf5 import *
from .fp_database_sqlite import *
from .fp_database_csv import *

        