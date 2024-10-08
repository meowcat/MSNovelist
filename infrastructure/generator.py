# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 23:45:42 2018

@author: dvizard
"""

import numpy as np
import smiles_process as sp
import tokens_process as tkp
import tensorflow as tf
import tensorflow.strings as tf_strings
import pickle
import warnings
import logging
logger = logging.getLogger("MSNovelist")

@tf.function
def fp_pipeline_map(fp, fp_map):
    return tf.gather(fp, fp_map, axis=1)

def mf_pipeline(mf):
    '''
    Extract molecular formula from dictionary form into array
    '''
    data = np.array(
        [[co[el] for el in sp.ELEMENTS_RDKIT] for co in mf])
    return data
    
def xy_tokens_pipeline(dataset_batch_smiles,
                length = tkp.SEQUENCE_LEN,
                initial = tkp.INITIAL_CHAR,
                final = tkp.FINAL_CHAR,
                pad = tkp.PAD_CHAR,
                embed_X = True):
    
    # Tokenize SMILES
    dataset_batch_Xy = dataset_batch_smiles.map(
        lambda s: tkp.tokens_to_tf(s))
    
    # One-hot encode
    dataset_batch_Xy = dataset_batch_Xy.map(
        lambda s: tkp.tokens_onehot(s))
    
    # Create X, sequence [0:n-1) and y, sequence [1:n)
    dataset_batch_Xy = dataset_batch_Xy.map(
        lambda s: tkp.tokens_split_xy(s))
    
    # extract X and y output to separate datasets
    dataset_batch_X = dataset_batch_Xy.map(lambda X, y: X)
    dataset_batch_y = dataset_batch_Xy.map(lambda X, y: y)
    
    # Add the grammar and element encoding to the input matrix
    # TODO: make sure i can remove this, since this is now done directly 
    # in the model.
    if embed_X:
        logger.info("using embed_X")
        dataset_batch_X = dataset_batch_X.map(
            lambda s: tkp.tokens_embed_all(s))
    else:
        logger.info("not using embed_X")
    return dataset_batch_X, dataset_batch_y

fpr_dict = {}            
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
                    unpackbits = False,
                    unpickle_mf = True,
                    embed_X = True,
                    map_fingerprints = True,
                    degraded_fingerprint_type = "float32",
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
    fprd_type = degraded_fingerprint_type
    try:
        fprd = tf.convert_to_tensor([row["fingerprint_degraded"] for row in dataset])
    except ValueError:
        warnings.warn("Degraded fingerprints not in dataset, using regular fingerprints")
        fprd_type = "uint8"
        fprd = fpr
    
    if unpickle_mf:
        logger.info("using unpickle_mf")
        mf = mf_pipeline([pickle.loads(row["mf"]) for row in dataset])
    else:
        mf = mf_pipeline([row["mf"] for row in dataset])
        logger.info("not using unpickle_mf")

    # Create datasets, zip, batch, unzip
    # This means: out of five separate datasets, we create
    # a single dataset, set the batch size, and extract the individual
    # batched sub-datasets again. We could also set the batch size
    # on each dataset independently, but this feels cleaner :)
    dataset_base = tf.data.Dataset.from_tensor_slices({
            'smiles_canonical': smiles_canonical,
            'smiles_generic': smiles_generic, 
            'fpr': fpr, 
            'fprd': fprd, 
            'mf' : mf
        })
    dataset_batch = dataset_base.batch(batch_size)
    dataset_batch_smiles_generic = dataset_batch.map(
        lambda ds: ds['smiles_generic'])
    dataset_batch_smiles_canonical = dataset_batch.map(
        lambda ds: ds['smiles_canonical'])
    dataset_batch_fpr = dataset_batch.map(lambda ds: ds["fpr"])
    dataset_batch_fprd = dataset_batch.map(lambda ds: ds["fprd"])
    dataset_batch_mf = dataset_batch.map(lambda ds: ds["mf"])

    
    # Fingerprint: extract and map
    # Unpack byte array (stored in database blob) to matrix

    def decode_set_shape(x, dtype, length):
        t = tf.io.decode_raw(x, dtype)
        t.set_shape([None, length])
        return t

    def decode_unpackbits(x, length,  bitorder="little"):
        if bitorder == "little":
            b = tf.constant((1, 2, 4, 8, 16, 32, 64, 128), shape=(1,1,8), dtype=tf.uint8)
        else:
            b = tf.constant((128, 64, 32, 16, 8, 4, 2, 1), shape=(1,1,8), dtype=tf.uint8)
        x_bitdim = tf.expand_dims(x, 2)
        bits = tf.bitwise.bitwise_and(x_bitdim, b) // b
        sh = tf.shape(bits)
        bits_wide = tf.reshape(bits, [-1, length * 8])
        #bits_wide.set_shape([None, length * 8])
        return bits_wide

    if unpack or unpackbits:
        fingerprint_len = tf_strings.length(fpr[0], unit="BYTE")
        print(f"Using data-derived fingerprint length of {fingerprint_len} (before bit-unpacking)")

    if unpack:
        logger.info("using unpack")
        dataset_batch_fpr = dataset_batch_fpr.map(
            lambda x: decode_set_shape(x, 'uint8', fingerprint_len))
        dataset_batch_fprd = dataset_batch_fprd.map(
            lambda x: decode_set_shape(x, fprd_type, fingerprint_len))
    else:
        logger.info("not using unpack")

    if unpackbits:
        logger.info("using unpackbits")
        dataset_batch_fpr = dataset_batch_fpr.map(
            lambda x: decode_unpackbits(x, fingerprint_len)
        )
        dataset_batch_fprd = dataset_batch_fprd.map(
            lambda x: decode_unpackbits(x, fingerprint_len)
        )

    # # Set a fixed size for the two outputs, otherwise decode_raw can't set the shape
    # # This is required because fixed_length= is buggy in our old version of TF
    # dataset_batch_fpr = dataset_batch_fpr.map(lambda x: x.set_shape([None, fingerprint_len]))
    # dataset_batch_fprd = dataset_batch_fprd.map(lambda x: x.set_shape([None, fingerprint_len]))

    # If required, map the full fingerprint to the CSI:FingerID-predicted subfingerprint. 
    # TODO: make sure we don't need this anymore and remove.
    # For training and evaulation, we only have the subfingerprint (6000),
    # for prediction we can easily do this earlier.
    if fp_map is not None:
        logger.info("using fp_map")
        fp_map_tensor = tf.convert_to_tensor(fp_map)
        dataset_batch_fpr = dataset_batch_fpr.map(lambda x: fp_pipeline_map(x, fp_map_tensor))
        dataset_batch_fprd = dataset_batch_fprd.map(lambda x: fp_pipeline_map(x, fp_map_tensor))
    else:
        logger.info("not using fp_map")
    # convert to float
    dataset_batch_fpr  = dataset_batch_fpr.map(lambda x: tf.cast(x, "float"))
    dataset_batch_fprd  = dataset_batch_fprd.map(lambda x: tf.cast(x, "float"))
    # MF is done, just convert to float
    dataset_batch_mf  = dataset_batch_mf.map(lambda x: tf.cast(x, "float"))
    # Hydrogens is just a subset of MF
    dataset_batch_h = dataset_batch_mf.map(lambda x:
                                           x[:,sp.ELEMENTS_RDKIT.index('H')])
    # On the SMILES side: map elements, pad, and split to X,y
    
    # Tokens processing
    dataset_tokens_X, dataset_tokens_y = xy_tokens_pipeline(
        dataset_batch_smiles_canonical, embed_X = embed_X)
    
    dataset = {'fingerprint': dataset_batch_fpr, 
            'fingerprint_degraded': dataset_batch_fprd, 
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
