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

from .base_fingerprinting import BaseFingerprinter

class SiriusFingerprinter(BaseFingerprinter):


    
    instance = None

    @classmethod
    def shutdown():
        pass

    @classmethod
    def static_fp_len(cls):
        return cls.instance.fp_map.fp_len

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            raise ValueError("Fingerprint instance not yet initialized")
        return cls.instance
    
    @classmethod
    def init_instance(cls, 
                      normalizer_path, sirius_path,
                      fp_map,  threads=1, capture=True, cache = None):
        cls.instance = cls(normalizer_path, sirius_path, fp_map, threads, capture, cache=cache)
    
    def __init__(self, lib_path, threads = 1, capture = True,cache=None):
        super.__init__(cache=cache)
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
        smiles_stdin = '\n'.join(smiles) + '\n'

        res_smiles  = subprocess.run(
            [ self.normalizer_bin ],
            input=smiles_stdin.encode('UTF-8'),
            capture_output=True,
            check=False
        )
        smiles_out = res_smiles.stdout.decode('UTF-8').rstrip('\n').split('\n')
        
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

            smiles_tempfile = tempfile.NamedTemporaryFile(mode = 'w', delete=False)
            smiles_tempfile.writelines(smiles_ok)
            smiles_tempfile.close()
            out_tempfile = tempfile.NamedTemporaryFile(delete=False)
            out_tempfile.close()


            res_fp  = subprocess.run(
                [ self.sirius_bin,
                  f"-i={smiles_tempfile.name}", 
                  "fingerprinter",
                  f"--output={out_tempfile.name}",
                  "--charge=0"    
                ],
                capture_output=True,
                check=False
            )

            def parse_fp(line):
                fp_parts = line.strip('\n').split('\t')
                fp_bits = fp_parts[1].split(',')
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

            with open(out_tempfile.name, 'r', encoding="UTF-8") as f:
                fp_out_ = f.readlines()
                fp_out = [ s.rstrip("\n") for s in fp_out_ ]
            #breakpoint()

            fp_by_id = { id: parse_fp(line) for id, line in zip(id_ok, fp_out) }

            for item in smiles_parsed:
                item['fingerprint'] = fp_by_id.get(item["data_id"], None)

        return smiles_parsed
    
    def get_fp_length(self):
        return self.static_fp_len
    
    def fingerprint_file(self, cores, file_in, file_out):
        raise NotImplementedError("This function was not yet implemented for the S6 fingerprinter.")

class Fingerprinter(SiriusFingerprinter):
    pass






# Functionality to test FP processing: alignment etc
# Todo: Make real unit tests

