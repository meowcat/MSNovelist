

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
from rdkit import Chem
from rdkit.Chem import AllChem

from .base_fingerprinting import BaseFingerprinter

    # def _get_morgan_fp_base(self, mol: data.Mol, nbits: int = 2048, radius=2):
    #     """get morgan fingeprprint"""

    #     def fp_fn(m):
    #         return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

    #     mol = mol.get_rdkit_mol()
    #     fingerprint = fp_fn(mol)
    #     array = np.zeros((0,), dtype=np.int8)
    #     DataStructs.ConvertToNumpyArray(fingerprint, array)
    #     return array

class MistFingerprinter(BaseFingerprinter):


    @classmethod
    def shutdown():
        pass
    
    instance = None
    bits = 4096
    radius = 2

    @classmethod
    def static_fp_len(cls):
        return cls.instance.bits

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            raise ValueError("Fingerprint instance not yet initialized")
        return cls.instance
    
    @classmethod
    def init_instance(cls, bits = 4096, radius = 2, cache = None):
        cls.instance = cls(bits, radius, cache = cache)
    
    def __init__(self, bits, radius, cache = None):
        super().__init__(cache = cache)
        self.bits = bits
        self.radius = radius
        
        
            
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

        def try_molfromsmiles(s, id = 0):
            m = None
            m = Chem.MolFromSmiles(s) 
            if m is None: 
                print(f"failed parsing id {id} - {s}")
            return m
        
        mol = [try_molfromsmiles(x, id) for id, x in enumerate(smiles)]
        def try_moltosmiles(m):
            smiles = None
            try:
                smiles = Chem.MolToSmiles(m, isomericSmiles=False)
            except:
                print("Error generating SMILES")
            return smiles
            
        smiles_kekulized = [
            try_moltosmiles(x) if x is not None
            else "" for  x in mol
            ]
        data = [{
            'data_id': i,
            'smiles_generic': x,
            'smiles_canonical': x
        } for i, x in enumerate(smiles_kekulized)]
        
        data_ok = [x for x in data if x["smiles_generic"] != ""]
        
        if calc_fingerprint:

            def calc_fp(smiles):

                fp = np.zeros((1, self.bits), dtype=np.uint8)
                try:
                    m = Chem.MolFromSmiles(smiles)
                    r = AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.bits)
                    fp_bits_num = [x for x in r.GetOnBits()]
                    fp[0,fp_bits_num] = 1
                except:
                    print("Error generating fingerprint")
                
                return fp
            
            def calc_and_pack_fp(smiles):
                fp = calc_fp(smiles)
                if return_numpy:
                    return fp
                fp_bytes = np.packbits(fp, bitorder='little').tobytes()
                if return_b64:
                    fp_b64 = base64.b64encode(fp_bytes)
                    return fp_b64
                return fp_bytes


            for x in data_ok:
                x["fingerprint"] = calc_and_pack_fp(x["smiles_generic"])

        return data_ok
    
    def get_fp_length(self):
        return self.bits

