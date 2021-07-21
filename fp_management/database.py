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


import h5py


# Below is auxiliary code that can hopefully go away in the future
import smiles_config as sc
import os
from tqdm import tqdm
import dill
import pickle

# db_pubchem_path = os.path.join(
#     sc.config["base_folder"],
#     "pubchem-cid-smiles-inchikey.db")

if 'db_pubchem' in sc.config:
    db_pubchem_path = sc.config["db_pubchem"]
else:
    db_pubchem_path = os.path.join(
        sc.config["base_folder"],
        "pubchem_ref/pubchem_ref.db")

db_pubchem = sqlite3.connect(db_pubchem_path)

import re

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


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
    
    

class FpItem:
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



class FpDatabase:
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
        ext_mapping = {
            ".csv": FpDatabaseCsv,
            ".db": FpDatabaseSqlite,
            ".pkl": fp_database_pickle,
            ".h5": FpDatabaseHdf5}
        db = ext_mapping[ext](db_path, db_config)
        return db
            

class FpDatabaseHdf5(FpDatabase):
    """
    HDF5 database backend. 
    Config options: 
        * process_smiles: if True, generic and canonical SMILES are generated by CDK.
            If False, the provided SMILES is used verbatim for both generic and canonical.
        * fingerprints_degraded: if a file is given, this is a set of pre-sampled fingerprints that
            are available as ['fingerprint_degraded'] in the records. 
            If None, there are no fingerprint_degraded provided.
        
    """
    
    
    def __init__(self, db_file, config):
        super().__init__(db_file, None) 
        
        config_ = {
            'use_regex': True,
            'fingerprints_degraded': None,
            'process_smiles': True
            }
        config_.update(config)
        self.config = config_
        self.fingerprinter = fpr.Fingerprinter.get_instance()
        self._load_hdf()
        self.use_regex = self.config['use_regex']

    def _load_hdf(self):
        self.db = h5py.File(self.db_file, 'r')
        # Load predefined sampled FPs if specified
        self.db_degraded = None
        if self.config["fingerprints_degraded"] is not None:
            fp_deg = self.config["fingerprints_degraded"]
            logger.info(f"Loading pre-sampled fingerprints from {fp_deg} into fingerprints_degraded")
            if "fingerprints_degraded" in self.db:
                logger.warn("Overriding the provided fingerprints_degraded from the database")
            self.db_degraded = h5py.File(fp_deg, 'r')
        # Process SMILES to aromatic SMILES if specified
        if self.config["process_smiles"]:
            logger.info("Generating aromatic SMILES")
            self.data = self.fingerprinter.process(self.db["smiles"], calc_fingerprint = False)
            self.smiles_generic = np.array([x["smiles_generic"] for x in self.data])
            self.smiles_canonical = np.array([x["smiles_canonical"] for x in self.data])
        else:
            logger.info("Not generating aromatic SMILES - using verbatim SMILES")
            self.smiles_generic = self.db["smiles"]
            self.smiles_canonical = self.db["smiles"]
        # Process molecular formulas
        logger.info("Processing molecular formulas (deferred)")
        mf_ = map(self._extract_mf, self.db["mf_text"])
        self.mf = np.array([x for x in mf_])
            
    def _extract_mf(self, mf_text):
        groups = re.findall(r'([A-Z][a-z]*)([0-9]*)', mf_text)
        mf_raw = dict(groups)
        mf = {
            k: max(1, int('0' + v)) for k, v in mf_raw.items()
            }
        return Counter(mf)
        
    def get_grp(self, grp):
        '''
        Parameters
        ----------
        grp : The group name to retrieve from the DB

        Returns
        -------
        An iterable 
            DESCRIPTION.

        '''
        
        if self.use_regex:
            grp_match = map(lambda s: re.match(grp, s) is not None, self.db["grp"])
        else:
            grp_match = map(lambda s: s.startswith(grp), self.db["grp"])
            
        grp_indices = np.where(np.array([x for x in grp_match]))[0]
        
        
        
        if self.db_degraded is not None:
            return [
                self._record_iter(x, grp)
                for x in zip(
                        grp_indices,
                        self.db["fingerprints"][grp_indices],
                        self.db_degraded["fingerprints_degraded"][grp_indices],
                        self.smiles_generic[grp_indices],
                        self.smiles_canonical[grp_indices],
                        self.db["inchikey"][grp_indices],
                        self.db["inchikey1"][grp_indices],
                        # for now we ignore that we have no processed molecules,
                        self.mf[grp_indices],
                        grp_indices)]
        elif "fingerprints_degraded" in self.db:
            return [
                self._record_iter(x, grp)
                for x in zip(
                        grp_indices,
                        self.db["fingerprints"][grp_indices],
                        self.db["fingerprints_degraded"][grp_indices],
                        self.smiles_generic[grp_indices],
                        self.smiles_canonical[grp_indices],
                        self.db["inchikey"][grp_indices],
                        self.db["inchikey1"][grp_indices],
                        # for now we ignore that we have no processed molecules,
                        self.mf[grp_indices],
                        grp_indices)]
        else:
            logger.warning("There are no fingerprints_degraded provided; the full fingerprints will be used for this!")
            return [
                self._record_iter(x, grp)
                for x in zip(
                        grp_indices,
                        self.db["fingerprints"][grp_indices],
                        self.db["fingerprints"][grp_indices],
                        self.smiles_generic[grp_indices],
                        self.smiles_canonical[grp_indices],
                        self.db["inchikey"][grp_indices],
                        self.db["inchikey1"][grp_indices],
                        # for now we ignore that we have no processed molecules,
                        self.mf[grp_indices],
                        grp_indices)]
                    
            
        
    def _record_iter(self, record, grp = None):
        keys = ["id", "fingerprint", "fingerprint_degraded", "smiles_generic", 
                "smiles_canonical", "inchikey", "inchikey1", "mf", "perm_order"]
        record_dict = dict(zip(keys, record))
        if grp is not None:
            record_dict["grp"] = grp
        return record_dict
        
    def get_pipeline_options(self):
        
        options = super().get_pipeline_options()
        options['unpack'] = False
        options['unpickle_mf'] = False
        return options


class FpDatabaseSqlite(FpDatabase):
    def __init__(self, db_file, config):
        super().__init__(db_file, None) 
        
        config_ = {
            'use_regex': True
            }
        config_.update(config)
        self.config = config_
        
        self.use_regex = self.config['use_regex']
        
        self._create_connection(db_file)
        self._create_schema()
    
    SCHEMA_DEF = {
            'compounds' :
                (
            ('id', 'INTEGER PRIMARY KEY'),
            ('fingerprint', 'BLOB'),
            ('fingerprint_degraded', 'BLOB'),
            ('smiles_generic', 'CHAR(128)'),
            ('smiles_canonical', 'CHAR(128)'),
            ('inchikey', 'CHAR(27)'),
            ('inchikey1', 'CHAR(14)'),
            ('mol', 'BLOB'),
            ('mf', 'BLOB'),
            ('mf_text', 'CHAR(128)'),
            ('source', 'CHAR(128)'),
            ('grp', 'CHAR(128)'),
            ('perm_order', 'INT')
                 )
                }
    
    # Copied from tutorial!
    def _create_connection(self, db_file):
        """ create a database connection to a SQLite database """
        try:
            self._db_con = sqlite3.connect(db_file)
            self._db_con.create_function("REGEXP", 2, regexp)
            self._db_con.row_factory = sqlite3.Row
        except Error as e:
            print(e)
            
    def _create_table(cls, conn, table_name, table_def):
        # build SQL query from table definition
        sql_str = 'CREATE TABLE IF NOT EXISTS '
        sql_str = sql_str + table_name + ' ('
        table_spec = [' '.join(table_tpl) for table_tpl in table_def]
        sql_str = sql_str + ', '.join(table_spec) + ' )'
        if conn is None:
            return(sql_str)
        else:
            try:
                c = conn.cursor()
                c.execute(sql_str)
            except Error as e:
                print(e)
    
    def _create_schema(self):
        for (table_name, table_schema) in FpDatabaseSqlite.SCHEMA_DEF.items():
            self._create_table(self._db_con, table_name, table_schema)
            
    def insert_fp(self, fp_item):
        sql_str = 'INSERT INTO compounds (smiles_generic, smiles_canonical, inchikey1, inchikey, fingerprint, fingerprint_degraded, mol, mf, mf_text, source, grp) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        sql_args = [fp_item.smiles_generic,
                    fp_item.smiles_canonical,
                    fp_item.inchikey1, fp_item.inchikey, 
                    fp_item.fp, 
                    fp_item.fp_degraded, 
                    sqlite3.Binary(pickle.dumps(fp_item.mol)),
                    sqlite3.Binary(pickle.dumps(fp_item.mf)),
                    fp_item.mf_text,
                     fp_item.source, fp_item.grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.lastrowid
        except Error as e:
            print(e)
    
    def insert_fp_multiple(self, fp_items):
        sql_str = 'INSERT INTO compounds (smiles_generic, smiles_canonical, inchikey1, inchikey, fingerprint, fingerprint_degraded, mol, mf, mf_text, source, grp) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                for fp_item in fp_items:
                    sql_args = [fp_item.smiles_generic,
                                fp_item.smiles_canonical,
                                fp_item.inchikey1, fp_item.inchikey, 
                                fp_item.fp, 
                                fp_item.fp_degraded, 
                                sqlite3.Binary(pickle.dumps(fp_item.mol)),
                                sqlite3.Binary(pickle.dumps(fp_item.mf)),
                                fp_item.mf_text,
                                fp_item.source, fp_item.grp]
                    cursor.execute(sql_str, sql_args)
            return cursor.lastrowid
        except Error as e:
            print(e)
    
    
    def close(self):
        self._db_con.close()
    
    def get_grp(self, grp):
        
        if self.use_regex:
            sql_str = "SELECT * from compounds where grp REGEXP ? order by perm_order"
        else:
            sql_str = 'SELECT * from compounds where grp = ? order by perm_order'
        sql_args = [grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
    
    def get_all(self):
        sql_str = 'SELECT * from compounds order by perm_order'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def get_by_mf(self, mf, grp):
        if self.use_regex:
            sql_str = 'SELECT * from compounds where mf_text = ? and grp REGEXP ? order by perm_order'
        else:
            sql_str = 'SELECT * from compounds where mf_text = ? and grp = ? order by perm_order'
        sql_args = [mf, grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
            
    def get_count_inchikey1(self, inchikey1, grp):
        sql_str = 'SELECT count() from compounds where inchikey1 = ? and grp = ?'
        sql_args = [inchikey1, grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchone()["count()"]
        except Error as e:
            print(e)

            
    def delete_grp(self, grp):
        sql_str = 'DELETE from compounds where grp = ?'
        sql_args = [grp]
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, sql_args)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def randomize(self, keep = True):
        sql_str = 'UPDATE compounds SET perm_order = random()'
        if keep:
            sql_str = 'UPDATE compounds SET perm_order = random() where perm_order IS NULL'
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
            
    def sql(self, sql_str):
        try:
            with self._db_con as con:
                cursor =  con.cursor()
                cursor.execute(sql_str)
            return cursor.fetchall()
        except Error as e:
            print(e)
        


def fp_database_pickle(db_file, config):
    return pickle.load(open(db_file, 'rb'))
    

class FpDatabaseCsv(FpDatabase):
    
    def __init__(self, db_file, config):
        # Update the default config with the given config:
        config_ = {
            'fp_map': None,
            'nrows': None,
            'construct_from': "inchi",
            "random_seed": 44,
            'reload_smiles_pubchem': False,
            'scramble_fingerprints': False
            }
        config_.update(config)
        super().__init__(db_file, config_)

        self._init_dispatch(config_['fp_map'],
                            config_['nrows'],
                            config_['construct_from'],
                            config_["random_seed"],
                            config_['reload_smiles_pubchem'],
                            config_['scramble_fingerprints'])
        
    def dump_pickle(self, path):
        self.fingerprinter = None
        pickle.dump(self, open(path, 'wb'))

    def _init_dispatch(self, fp_map, nrows = None, 
                       construct_from = "inchi", 
                       random_seed = 44,
                       reload_smiles_pubchem = False,
                       scramble_fingerprints = False):
        '''
        Parameters
        ----------
        db_file : TYPE
            DESCRIPTION.
        fp_map : int
            Note: this needs to be the "long" fingerprint map - 3xxx bits from
            CANOPUS / CSI:FID 4.4) which is in column 1 of the "statistics" csv
        nrows: int
            If given, limits how many records are read - for e.g. unittest
            purposes
        Returns
        -------
        None.

        '''
        self.data_grp = {}
        # read information block
        self.data_information = pd.read_table(
            self.db_file, delimiter='\t', 
            header=None,
            usecols = (0,1,2,3), 
            names=("id","inchikey", construct_from ,"fingerprint"), 
            nrows = nrows,
            comment = None)
        self.data_information.set_index("id", inplace=True)
        self.fp_map = fpm.FingerprintMap(fp_map)
        self.fp_len = len(self.data_information["fingerprint"][0])
        self.fp_real_len = int(max(self.fp_map.positions)+1)
        self.random_seed = random_seed
        
        # Read predicted fingerprints
        data_fp_predicted = np.genfromtxt(
            self.db_file, 
            delimiter='\t', 
            names=None, 
            comments = None,
            usecols= tuple(map(lambda x: x + 4,
                              range(self.fp_len)
                              )),
            max_rows = nrows
            )
        
        if reload_smiles_pubchem:
            self.data_information = get_smiles_pubchem(self.data_information, db_pubchem)
            self.data_information["smiles"] = self.data_information["smiles_in"]
            
        
        self.fingerprinter = fpr.Fingerprinter.get_instance()
        
        self.construct_from = construct_from
        # Read real fingerprints
        data_fp_true = np.array([list(map(int, x)) 
                                 for x in self.data_information["fingerprint"]])

        # Reshape predicted 3541 FP into the full 7593 bit FP
        data_fp_full = np.zeros((data_fp_predicted.shape[0],
                                 self.fp_real_len))
        data_fp_full[:,self.fp_map.positions] = data_fp_predicted
        # data_fp_realigned = np.array([
        #     fpr.realign_fp_numpy(i) for i in list(data_fp_full)])
        self.data_fp_predicted = data_fp_full
        # Reshape true 3541 FP into the full 7593 bit FP
        data_fp_full = np.zeros((data_fp_true.shape[0],
                                 self.fp_real_len))
        data_fp_full[:,self.fp_map.positions] = data_fp_true
        # data_fp_realigned = np.array([
        #     fpr.realign_fp_numpy(i) for i in list(data_fp_full)])
        self.data_fp_true = data_fp_full
        
        
        self.data_information["perm_order"] = 0
        self.data_information["source"] = ''
        self.data_information["grp"] = ''
        self.data_information["row_id"] = np.arange(
            len(self.data_information), dtype="int32")
        
        # For now, we can only scramble predicted fingerprints,
        # because we need the true ones for comparison, right?
        # Or are those pulled from the fingerprinter?
        
        self.scramble_fingerprints = scramble_fingerprints
        if self.scramble_fingerprints:
            fp_order = np.arange(data_fp_predicted.shape[0])
            np.random.shuffle(fp_order)
            data_fp_predicted = data_fp_predicted[fp_order,:]
            self.data_fp_predicted = data_fp_predicted
            
        self.process_smiles()
        self.randomize(self.random_seed)

    def process_smiles(self):
        
        self.data_information = process_df(
            self.data_information,
            self.fingerprinter,
            construct_from = self.construct_from,
            write = ["smiles_generic", "smiles_canonical", "inchikey", "inchikey1","mf", "mol"],
            block_id = None)
            
    def close(self):
        pass
    
    def set_grp(self, name, table, fold = False):
        '''
        

        Parameters
        ----------
        name : Name of the group to be created
            DESCRIPTION.
        table : Pandas dataframe with an "id" column - or the first column
            will be used - to select the database items to be assigned to
            this group.
        fold: optional
            If True, then not one group but n groups will be created, with
            the name as a prefix and the fold id - derived from a "fold" 
            column in the 'table' - as a suffix.
        Returns
        -------
        None.

        '''
        if not any(table.columns == "id"):
            table["id"] = table.iloc[:,0]
        if not fold:
            self.data_grp.update({
                    name: table["id"]
                })
            # Verify that the group is completely present,
            # otherwise this will fail:
            #_ = self.get_grp(name)
        else:
            # Make folds
            folds = set(table["fold"])
            for fold in folds:
                self.data_grp.update({
                    name + str(fold): table.loc[table["fold"] == fold]
                    })
                #_ = self.get_grp(name + str(fold))
    
    
    def get_grp(self, grp):
        '''
        untested

        Parameters
        ----------
        grp : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if grp in self.data_grp:
            grp_ok = self.data_information.reindex(self.data_grp[grp])
            missing = grp_ok["row_id"].isna()
            grp_ok = grp_ok.copy()
            grp_ok["grp"] = grp
            if any(missing):
                n_missing = sum(missing)
                warn(f"Not all entries from this group are present: {n_missing} missing")
        else:
            grp_match = map(lambda s: s.startswith(grp), self.data_information.index)
            grp_ok = self.data_information.loc[grp_match].copy()
            grp_ok["grp"] = grp
            

        
        #return grp_ok
        return [
            self._record_iter(x)
            for x in grp_ok.loc[grp_ok["row_id"].notna()].sort_values("perm_order").itertuples()
            ]
            
    def get_all(self):
        return [self._record_iter(x) 
                for x in 
                self.data_information.sort_values("perm_order").itertuples()]
        
    def _record_iter(self, record):
        d = record._asdict()
        row_id = int(d["row_id"]) # this is very ugly, because
        # row_id gets transformed to float during reindex() in get_grp()
        # when NA are introduced
        #print(row_id)
        d.update({
            'fingerprint': self.data_fp_true[row_id,:],
            "fingerprint_degraded": self.data_fp_predicted[row_id,:]
            })
        return d
                   
    def randomize(self, keep = True, random_seed = 45):
        if random_seed is not None:
            seed(random_seed)
        self.data_information["perm_order"] = \
            [random() for _ in range(self.data_information.shape[0])]
            
    def get_pipeline_options(self):
        
        options = super().get_pipeline_options()
        options['unpack'] = False
        options['unpickle_mf'] = False
        return options
            
        