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


try:
    import jpype as jp
    _java_enabled = True
except:
    _java_enabled = False
    
class Fingerprinter:
    @classmethod
    def shutdown():
        jp.shutdownJVM()
    
    instance = None

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            raise ValueError("Fingerprint instance not yet initialized")
        return cls.instance
    
    @classmethod
    def init_instance(cls, lib_path, threads=1, capture=True, cache = None):
        cls.instance = cls(lib_path, threads, capture, cache)
    
    def __init__(self, lib_path, threads = 1, capture = True, cache = None):
        self.threads = threads
        java_mem = sc.config['java_memory']
        option_xmx = f"-Xmx{java_mem}m"
        option_xms = f"-Xms{java_mem}m"
        if not jp.isJVMStarted():
            jp.startJVM(jp.getDefaultJVMPath(), 
                        option_xmx, option_xms, "-Djava.class.path="+lib_path,
                        convertStrings = True)
        fpu = jp.JClass('ch.moduled.fingerprintwrapper.FingerprintUtil').instance
        self.n_fingerprinters = fpu.makeFingerprinters(2*threads)
        # Setup logging from Java
        self.capture = capture
        if capture:
            mystream = jp.JProxy("ch.moduled.fingerprintwrapper.IPythonPipe", inst=sys.stdout)
            errstream = jp.JProxy("ch.moduled.fingerprintwrapper.IPythonPipe", inst=sys.stderr)
            outputstream = jp.JClass("ch.moduled.fingerprintwrapper.PythonOutputStream")()
            outputstream.setPythonStdout(mystream)
            ps = jp.JClass("java.io.PrintStream")
            err_stream = jp.JClass("ch.moduled.fingerprintwrapper.PythonOutputStream")()
            err_stream.setPythonStdout(errstream)
            jp.java.lang.System.setOut(ps(outputstream, True))
            jp.java.lang.System.setErr(ps(err_stream, True))

        self.fpu = fpu
        
        self.cache_path = cache
        self.cache = None
        self.cache_connect()
        
    def cache_connect(self):
        if self.cache_path is None:
            self.cache = None
            return
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
            
    def process(self, smiles, calc_fingerprint = True, return_b64 = True):
        '''
        Raw fingerprinting (not cache-aware) directly from a list of SMILES

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        calc_fingerprint : TYPE, optional
            DESCRIPTION. The default is True.
        return_b64 : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        res : TYPE
            DESCRIPTION.

        '''        
        res = self.fpu.processSmilesFromPython(
            smiles,
            self.threads,
            calc_fingerprint,
            return_b64,
            self.capture
            )
        if calc_fingerprint:
            res = [{
                "data_id": entry[0],
                "smiles_generic": entry[1],
                "smiles_canonical": entry[2],
                "fingerprint": entry[3]
                } for entry in res]
        else:
            res = [{
                "data_id": entry[0],
                "smiles_generic": entry[1],
                "smiles_canonical": entry[2],
                } for entry in res]
        res.sort(key=lambda x: x["data_id"])
        return res
    def fingerprint_file(self, cores, file_in, file_out):
        self.fpu.process(file_in, file_out, cores)


# B64-decodes one fingerprint
def get_fp(fp, length = 8925, b64decode = True):
    if fp is None:
        return None
    if b64decode:
        fp_bytes = base64.b64decode(fp)
    else:
        fp_bytes = fp
    fp_bytes = np.frombuffer(fp_bytes, dtype=np.uint8).reshape(1, -1)
    fp_bits = process_fp_numpy_block(fp_bytes)
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
        