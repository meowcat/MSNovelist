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





class Fingerprinter:
    @classmethod
    def shutdown():
        pass
    
    instance = None

    @classmethod
    def static_fp_len(cls):
        return cls.instance.fp_map.fp_len

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            raise ValueError("Fingerprint instance not yet initialized")
        return cls.instance
    
    @classmethod
    def init_instance(cls, lib_path, fp_map,  threads=1, capture=True, cache = None):
        cls.instance = cls(lib_path, fp_map, threads, capture, cache)
    
    def __init__(self, lib_path, fp_map, threads = 1, capture = True, cache = None,
                 ):
        self.threads = threads
        self.lib_path = lib_path
        self.fingerprinter_bin = pathlib.Path(lib_path) / "fingerprinter_cli"
        self.smiles_normalizer_bin = pathlib.Path(lib_path) / "smiles_normalizer"
        self.fp_map = fp_map
        self.fp_len = fp_map.fp_len

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
            fingerprints = [get_fp(x["fingerprint"]) for x in fingerprints_]
            self.cache_add(data_nohit["inchikey1"], fingerprints)
        else:
            fingerprints_ = self.process(data_[in_column].fillna("").to_list())
            fingerprints = [get_fp(x["fingerprint"]) for x in fingerprints_]


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
            
    def process(self, smiles, calc_fingerprint = True, 
                return_b64 = True, return_numpy = False):
        '''
        Raw fingerprinting (not cache-aware) directly from a list of SMILES. ALso
        performs SMILES canonicalization according to the format needed for 
        MSNovelist training and processing.

        To only run SMILES normalization, use `calc_fingerprint = False`. This is
        much faster.

        Note: in pre-SIRIUS6 MSNovelist, this called a wrapper lib in Java via Jpype
        which performed SMILES canonicalization and fingerprinting. In SIRIUS6
        MSNovelist, canonicalization and fingerprinting are performed by two independent
        CLI tools (`fingerprinter_cli` for fingerprinting, 
        `smiles_normalizer` for SMILES processing).

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        calc_fingerprint : bool, optional
            Whether to perform the (slow) fingerprint calculation. The default is True.
        return_b64 : bool, optional
            Whether to return b64-encoded form of fingerprint. The default is True.
        return_numpy: bool, optional
            Whether to return the fingerprint as a Numpy array. The default is False.
            This is to prepare a refactoring where we get rid of the legacy b64 and byte 
            return modes. The new way of calculating gives us the array as desired directly,
            so we should get rid of the other modes once we fix all dependencies in the
            rest of the code.

        Returns
        -------
        res : list[dict]
            Returns a list of dictionaries with elements:
            `data_id`:  the index of the entry in the input list (see below)
            `smiles_generic`: The SMILES in aromatic, but not canonicalized form according to CDK
            `smiles_canonical`: The SMILES in aromatic, canonicalized form according to CDK
            `fingerprint`: the calculated fingerprint. For backwards compatibility,
                this returns the base64-encoded fingerprint by default and a byte array
                if `return_b64` is False, because in the old version the result was returned
                from Java as byte array. Now we are artifically recreating these structures
                to avoid changes in the rest of the code. This should be 

        '''
        smiles_stdin = '\n'.join(smiles)

        res_smiles  = subprocess.run(
            [ self.smiles_normalizer_bin ],
            input=smiles_stdin.encode('UTF-8'),
            capture_output=True,
            check=False
        )
        smiles_out = res_smiles.stdout.decode('UTF-8').rstrip('\n').split('\n')
        #print(smiles_out)
        
        def parse_line(id, line):
            line_ = line.split('\t')
            if line_[0] == "OK":
                return {
                    'data_id': id,
                    'smiles_generic': line_[1],
                    'smiles_canonical': line_[2]
                }
            return {
                'data_id': id,
                'smiles_generic': "",
                'smiles_canonical': ""
            }

        smiles_parsed = [parse_line(id, line) for id, line in enumerate(smiles_out)]
        id_ok = [x["data_id"] for x in smiles_parsed if x["smiles_generic"] != ""]
        smiles_ok = [smiles[x] for x in id_ok]
        
        if calc_fingerprint:

            smiles_fp_stdin = '\n'.join(smiles_ok)

            res_fp  = subprocess.run(
                [ self.fingerprinter_bin ],
                input = smiles_fp_stdin.encode('UTF-8'),
                capture_output=True,
                check=False
            )

            def parse_fp(line):
                fp_bits = line.strip('\n').split('\t')
                fp_bits_num_ = [int(x) for x in fp_bits]
                fp_bits_num = [x for x in fp_bits_num_ if x <  self.fp_len]
                fp = np.zeros((1, self.fp_len), dtype=np.uint8)
                fp[0,fp_bits_num] = 1
                if return_numpy:
                    return fp
                fp_bytes = np.packbits(fp, bitorder='little').tobytes()
                if return_b64:
                    fp_b64 = base64.b64encode(fp_bytes)
                    return fp_b64
                return fp_bytes

            fp_out = res_fp.stdout.decode('UTF-8').rstrip('\n').split('\n')
            #breakpoint()

            fp_by_id = { id: parse_fp(line) for id, line in zip(id_ok, fp_out) }

            for item in smiles_parsed:
                item['fingerprint'] = fp_by_id.get(item["data_id"], None)

        return smiles_parsed
    
    def fingerprint_file(self, cores, file_in, file_out):
        raise NotImplementedError("This function was not yet implemented for the S6 fingerprinter.")

# B64-decodes one fingerprint
def get_fp(fp, length = None, b64decode = True):
    if length is None:
        length = Fingerprinter.static_fp_len()
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
    return(fp_bits)

def process_fp_numpy_block(fp_bytes):
    # fp_block = np.r_[[np.frombuffer(fp, dtype=np.uint8) 
    #     for fp in fp_bytes if fp is not None]]
    fp_block = np.unpackbits(fp_bytes, axis = 1, bitorder="little")
    return fp_block

def repack_fp_numpy(X_fp):
    fp_bytes = np.packbits(X_fp, bitorder = 'little').tobytes()
    return fp_bytes



# Functionality to test FP processing: alignment etc
# Todo: Make real unit tests
        