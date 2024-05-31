# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:19:33 2018

@author: dvizard
"""


import numpy as np
import base64
from bitstring import BitArray, BitStream, ConstBitStream
import os
import sys
import smiles_config as sc
import sqlite3
from sqlite3 import Error
import pickle
from warnings import warn
import subprocess
import pathlib
import tempfile

class BaseFingerprinter:

    def __init__(self, cache = None):

        self.cache_path = cache
        self.cache = None
        self.cache_connect()
        
    def cache_connect(self):
        
        if self.cache_path is None:
            self.cache = None
            return
        
        if not pathlib.Path(self.cache_path).parent.exists():
            pathlib.Path(self.cache_path).parent.mkdir(parents=True)
        self.cache = sqlite3.connect(self.cache_path)
        with self.cache as con:
            cursor = con.cursor()
            cursor.execute(
                '''
                CREATE TABLE IF NOT EXISTS fingerprint_cache 
                           (inchikey1 CHAR(14) PRIMARY KEY, 
                            fingerprint BLOB)
                ''')
            cursor.execute(
                '''
                CREATE UNIQUE INDEX IF NOT EXISTS inchikey1_index
                    ON fingerprint_cache (inchikey1)
                ''')
        return
            
    def cache_query(self, inchikey1):
        sql_str = "SELECT fingerprint from fingerprint_cache WHERE inchikey1 = ?"
        def _cache_query(key):
            if key == "":
                return None
            #try:
            with self.cache as con:
                cursor =  con.cursor()
                cursor.execute(sql_str, [key])
                data = cursor.fetchall()
                if len(data) > 1:
                    warn("Duplicate entry in fingerprint cache")
                if len(data) < 1:
                    return None
                return pickle.loads(data[0][0])
            #except Error as e:
                #warn(e)
             #   return np.nan
        return [_cache_query(k) for k in inchikey1]
    
    def cache_add(self, inchikey1, fingerprint):
        sql_str = "INSERT OR IGNORE INTO fingerprint_cache (inchikey1, fingerprint) VALUES (?, ?)"
        keys_added = []
        def _cache_add(key, fp, con):
            if fp is None:
                return
            if key == "":
                return
            if key in keys_added:
                return
            keys_added.append(key)
            fp_blob = pickle.dumps(fp)
            # try:
            cursor = con.cursor()
            cursor.execute(sql_str, [key, fp_blob])
            # except Error as e:
            #     warn(e)
        with self.cache as con:
            for i, f in zip(inchikey1, fingerprint):
                _cache_add(i, f, con)
            con.commit()
            
    def process_df(self, data,
                   in_column = "smiles",
                   out_column = "fingerprint",
                   verbose_cache_column = None,
                   inplace = False):
        '''
        Cache-aware fingerprinting

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        # If using cache, query cache and fill up if necessary
        data_ = data.copy()
        if ("inchikey1" in data_.columns) and (self.cache is not None):
            data_[out_column] = self.cache_query(data_["inchikey1"])
            data_nohit = data_.loc[data_[out_column].isna()]
            fingerprints_ = self.process(data_nohit[in_column].fillna("").to_list())
            fingerprints = [self.get_fp(x["fingerprint"]) for x in fingerprints_]
            self.cache_add(data_nohit["inchikey1"], fingerprints)
        else:
            fingerprints_ = self.process(data_[in_column].fillna("").to_list())
            print(fingerprints_)
            fingerprints = [self.get_fp(x["fingerprint"]) for x in fingerprints_]


        if inplace:
            data_out = data
        else:
            data_out = data_
        
        # If using cache, reread results that are now in cache
        # into the out dataframe
        if self.cache is not None:    
            data_out[out_column] = self.cache_query(data["inchikey1"])
            if verbose_cache_column is not None:
                data_out[verbose_cache_column] = [key not in data_nohit["inchikey1"].tolist() 
                                               for key in data_out["inchikey1"].tolist()]
        # If not using cache, copy the fingerprints directly into the outcol
        # of the out dataframe
        else:
            data_out[out_column] = fingerprints
            if verbose_cache_column is not None:
                data_out[verbose_cache_column] = False

        if inplace:
            return
        return data_


    
    def get_fp(self, fp, length = None, b64decode = True):
        if length is None:
            length = self.get_fp_length()
        return get_fp(fp, length, b64decode)



# B64-decodes one fingerprint
def get_fp(fp, length, b64decode = True):
    if length is None:
        return None
    if fp is None:
        return None
    if b64decode:
        fp_bytes = base64.b64decode(fp)
    else:
        fp_bytes = fp
    fp_bytes = np.frombuffer(fp_bytes, dtype=np.uint8).reshape(1, -1)
    fp_bits = process_fp_numpy_block(fp_bytes)
    # potentially work around alignment issues on the last bits
    fp_bits = fp_bits[:,:length]
    return fp_bits

def process_fp_numpy_block(fp_bytes):
    # fp_block = np.r_[[np.frombuffer(fp, dtype=np.uint8) 
    #     for fp in fp_bytes if fp is not None]]
    fp_block = np.unpackbits(fp_bytes, axis = 1, bitorder="little")
    return fp_block

def repack_fp_numpy(X_fp):
    fp_bytes = np.packbits(X_fp, bitorder = 'little').tobytes()
    return fp_bytes