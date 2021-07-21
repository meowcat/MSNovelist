# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:45:42 2018

@author: dvizard
"""

import fp_management.fingerprinting as fp
import smiles_config as sc
import numpy as np
import smiles_process as sp
import tokens_process as tkp
import tensorflow as tf
import pickle
import warnings
import logging
logger = logging.getLogger("MSNovelist")

#from tensorflow import io


def fp_pipeline(fpd, fp_map = None, unpack = True):
    if unpack:
        fpd = np.frombuffer(fpd, dtype=np.uint8).reshape((fpd.shape[0], -1))
        fpd = fp.process_fp_numpy_block(fpd)
    if fp_map is not None:
        fpd = fpd[:,fp_map]
    return fpd

@tf.function
def fp_pipeline_unpack(fp):
    # unpack string into uint8 array
    fp_decoded = tf.io.decode_raw(fp, 'uint8')
    #fp_decoded = tf.reshape(fp_decoded, (tf.shape(fp)[0], -1))
    # unpack uint8 array into bits
    # https://stackoverflow.com/a/45459877/1259675
    # 
    # bits = tf.reshape(
    #     tf.constant((128, 64, 32, 16, 8, 4, 2, 1), dtype=tf.uint8),
    #     (1, 1, -1)
    #     )
    bits = tf.reshape(
        tf.constant((1, 2, 4, 8, 16, 32, 64, 128), dtype=tf.uint8),
        (1, 1, -1))
    fp_decoded = tf.expand_dims(fp_decoded, 2)
    fp_unpacked = tf.reshape(
        tf.bitwise.bitwise_and(fp_decoded, bits), 
        (tf.shape(fp)[0], -1))
    fp_unpacked = tf.cast(fp_unpacked != 0, 'uint8')
    return fp_unpacked

@tf.function
def fp_pipeline_map(fp, fp_map):
    return tf.gather(fp, fp_map, axis=1)


def mf_pipeline(mf):
    data = np.array(
        [[co[el] for el in sp.ELEMENTS_RDKIT] for co in mf])
    return data
       
    
def xy_pipeline(dataset_batch_smiles,
                length = sp.SEQUENCE_LEN,
                initial = sp.INITIAL_CHAR,
                final = sp.FINAL_CHAR,
                pad = sp.TF_PAD_CHAR,
                mol_form = None,
                hinting = False):
    dataset_batch_Xy = dataset_batch_smiles.\
        map(lambda s: sp.smiles_map_elements(s))
    dataset_batch_Xy = dataset_batch_Xy.\
        map(lambda s: sp.smiles_pad(s, length, initial, final, pad, "back"))
    dataset_batch_Xy = dataset_batch_Xy.\
        map(lambda s: sp.smiles_xysplit(s))
    dataset_batch_X = dataset_batch_Xy.map(lambda X, y: X)
    dataset_batch_y = dataset_batch_Xy.map(lambda X, y: y)
    # X dataset: encode one-hot, then add hinting
    dataset_batch_X = dataset_batch_X.\
        map(lambda s: sp.smiles_encode_x(s, sp.TF_PAD_CHAR))
    # Add hinting in preprocessing?
    # Since we now moved hinting to the model with RAC layers, this is 
    # not in use anymore.
    if hinting:
        dataset_batch_X = dataset_batch_X.\
            map(lambda X: sp.smiles_attach_hints(
                X_smiles = X, mol_form=mol_form,
                formula_hint=True,
                grammar_hint=True))
    # y dataset: encode one-hot and be done
    dataset_batch_y = dataset_batch_y.\
        map(sp.smiles_encode_y)
    # Convert all to float
    dataset_batch_X  = dataset_batch_X.map(lambda x: tf.cast(x, "float"))
    dataset_batch_y  = dataset_batch_y.map(lambda x: tf.cast(x, "float"))
    return dataset_batch_X, dataset_batch_y
    
def xy_tokens_pipeline(dataset_batch_smiles,
                length = tkp.SEQUENCE_LEN,
                initial = tkp.INITIAL_CHAR,
                final = tkp.FINAL_CHAR,
                pad = tkp.PAD_CHAR,
                embed_X = True):
    dataset_batch_Xy = dataset_batch_smiles.map(
        lambda s: tkp.tokens_to_tf(s))
    dataset_batch_Xy = dataset_batch_Xy.map(
        lambda s: tkp.tokens_onehot(s))
    dataset_batch_Xy = dataset_batch_Xy.map(
        lambda s: tkp.tokens_split_xy(s))
    dataset_batch_X = dataset_batch_Xy.map(lambda X, y: X)
    dataset_batch_y = dataset_batch_Xy.map(lambda X, y: y)
    
    if embed_X:
        dataset_batch_X = dataset_batch_X.map(
            lambda s: tkp.tokens_embed_all(s))
    return dataset_batch_X, dataset_batch_y
            

def smiles_pipeline(dataset, 
                    batch_size,
                    fp_map = None,
                    length = sp.SEQUENCE_LEN,
                    initial = sp.INITIAL_CHAR,
                    final = sp.FINAL_CHAR,
                    pad = sp.TF_PAD_CHAR,
                    mol_form = None,
                    hinting = False,
                    return_smiles = False,
                    unpack = True,
                    unpickle_mf = True,
                    embed_X = True,
                    map_fingerprints = True,
                    **kwargs):
    
    if not map_fingerprints:
        fp_map = None
    
    smiles_canonical, smiles_generic = map(
        np.array,
        zip(*[
            (row["smiles_canonical"], row["smiles_generic"]) 
            for row in dataset
            ]))
    
    fpr = tf.convert_to_tensor([row["fingerprint"] for row in dataset])
    try:
        fprd = tf.convert_to_tensor([row["fingerprint_degraded"] for row in dataset])
    except ValueError:
        warnings.warn("Degraded fingerprints not in dataset, using regular fingerprints")
        fprd = fpr
    
    if unpickle_mf:
        mf = mf_pipeline([pickle.loads(row["mf"]) for row in dataset])
    else:
        mf = mf_pipeline([row["mf"] for row in dataset])
    # # On the fingerprint side: Transform the fingerprints offline, because
    # # it is small and fast enough, and we don't need to convert all functions
    # # to TF.
    # fpr = fp_pipeline(fpr, fp_map, unpack = unpack)
    # try:
    #     fprd = fp_pipeline(fprd, fp_map, unpack = unpack)
    # except IndexError:
    #     warnings.warn("Degraded fingerprints are not available")
    #     fprd = np.zeros_like(fpr)

    # Create datasets, zip, batch, unzip
    dataset_base = tf.data.Dataset.from_tensor_slices(
        {
            'smiles_canonical': smiles_canonical,
            'smiles_generic': smiles_generic, 
            'fpr': fpr, 
            'fprd': fprd, 
            'mf' : mf}
        )
    dataset_batch = dataset_base.batch(batch_size)
    dataset_batch_smiles_generic = dataset_batch.map(
        lambda ds: ds['smiles_generic'])
    dataset_batch_smiles_canonical = dataset_batch.map(
        lambda ds: ds['smiles_canonical'])
    dataset_batch_fpr = dataset_batch.map(lambda ds: ds["fpr"])
    dataset_batch_fprd = dataset_batch.map(lambda ds: ds["fprd"])
    dataset_batch_mf = dataset_batch.map(lambda ds: ds["mf"])

    
    # Fingerprint: extract and map
    if unpack:
        dataset_batch_fpr = dataset_batch_fpr.map(fp_pipeline_unpack)
        dataset_batch_fprd = dataset_batch_fprd.map(fp_pipeline_unpack)
    if fp_map is not None:
        fp_map_tensor = tf.convert_to_tensor(fp_map)
        dataset_batch_fpr = dataset_batch_fpr.map(lambda x: fp_pipeline_map(x, fp_map_tensor))
        dataset_batch_fprd = dataset_batch_fprd.map(lambda x: fp_pipeline_map(x, fp_map_tensor))
    # convert to float
    dataset_batch_fpr  = dataset_batch_fpr.map(lambda x: tf.cast(x, "float"))
    dataset_batch_fprd  = dataset_batch_fprd.map(lambda x: tf.cast(x, "float"))
    # MF is done, just convert to float
    dataset_batch_mf  = dataset_batch_mf.map(lambda x: tf.cast(x, "float"))
    # Hydrogens is just a subset of MF
    dataset_batch_h = dataset_batch_mf.map(lambda x:
                                           x[:,sp.ELEMENTS_RDKIT.index('H')])
    # On the SMILES side: map elements, pad, and split to X,y
    dataset_batch_X_generic, dataset_batch_y_generic = xy_pipeline(
        dataset_batch_smiles_generic,
        length, initial, final, pad, mol_form,
        hinting
        )
    dataset_batch_X_canonical, dataset_batch_y_canonical = xy_pipeline(
        dataset_batch_smiles_generic,
        length, initial, final, pad, mol_form,
        hinting
        )
    
    # Tokens processing
    dataset_tokens_X, dataset_tokens_y = xy_tokens_pipeline(
        dataset_batch_smiles_canonical, embed_X = embed_X)
    
    dataset = {'fingerprint': dataset_batch_fpr, 
            'fingerprint_degraded': dataset_batch_fprd, 
            'smiles_X_generic': dataset_batch_X_generic, 
            'smiles_y_generic': dataset_batch_y_generic, 
            'smiles_X_canonical': dataset_batch_X_canonical, 
            'smiles_y_canonical': dataset_batch_y_canonical, 
            'mol_form': dataset_batch_mf,
            'smiles_generic': dataset_batch_smiles_generic,
            'smiles_canonical': dataset_batch_smiles_canonical,
            'n_hydrogen': dataset_batch_h,
            'tokens_X': dataset_tokens_X,
            'tokens_y': dataset_tokens_y
            }
    return dataset
    
    
def dataset_zip(data, format_X = None, format_y = None, fingerprint_selected = None,
                **kwargs):
    if fingerprint_selected is not None:
        logger.info(f'Selecting fingerprint {fingerprint_selected}')
        data['fingerprint_selected'] = data[fingerprint_selected]
    if format_X is None:
        data_X = tf.data.Dataset.zip(data)
    else:
        data_X = tf.data.Dataset.zip({key: data[key] for key in format_X})
    if format_y is not None:
        data_y = tf.data.Dataset.zip(tuple(data[idx] for idx in format_y))
        return tf.data.Dataset.zip((data_X, data_y))
    else:
        return data_X


def dataset_blueprint(data):
    blueprint = {k : next(iter(d)).numpy() for k, d in data.items()}
    return(blueprint)
