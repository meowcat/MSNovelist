7# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:46:43 2020

@author: stravsm
"""


from .definitions import *
import selfies as sf
import numpy as np
import tensorflow as tf
from tensorflow import strings as ts



def ts_decode(s):
    '''
    Shorthand to get to Python strings from a TF byte/string tensor

    Parameters
    ----------
    s : tf.Tensor dtype=string

    Returns
    -------
    List of strings.

    '''
    return [x.decode() for x in s.numpy()]



def selfies_to_tf_wrap(smiles):
    selfies_ = selfies_from_smiles(ts_decode(smiles))
    selfies_ = selfies_pad(selfies_)
    selfies_mat = selfies_encode(selfies_)
    return selfies_mat

def selfies_to_tf(smiles):
    
    selfies_mat = tf.py_function(
        selfies_to_tf_wrap,
        [smiles],
        Tout = 'int32')
    selfies_mat.set_shape(tf.TensorShape([None, None]))
    return selfies_mat


def selfies_from_smiles(smiles):
    selfies = [list(sf.split_selfies(sf.encoder(s))) for s in smiles]
    return selfies
    
def selfies_to_smiles(selfies):
    smiles = [sf.decoder(s) for s in ts_decode(selfies)]
    return smiles
    

def selfies_pad_one(selfies, 
                length = SEQUENCE_LEN, 
                initial_char = INITIAL_CHAR, 
                final_char = FINAL_CHAR,
                pad_char = PAD_CHAR):
    selfies.insert(0, initial_char)
    selfies.append(final_char)
    selfies.extend([pad_char] * (length - len(selfies)))
    return selfies[:length]
    
def selfies_pad(selfies, 
                length = SEQUENCE_LEN, 
                initial_char = INITIAL_CHAR, 
                final_char = FINAL_CHAR,
                pad_char = PAD_CHAR):
    return [selfies_pad_one(x, length, initial_char, final_char, pad_char) for x in selfies]
    

def selfies_encode_one(selfies):
    return [SELFIES_MAP.get(c, 0) for c in selfies]

def selfies_encode(selfies):
    return tf.stack([selfies_encode_one(x) for x in selfies], axis=0)

def selfies_encode_mf(selfies_onehot_mat):
    '''
    Append element vector to the "raw" onehot SELFIES vector.

    Parameters
    ----------
    selfies_onehot_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return tf.concat([
        tf.matmul(selfies_onehot_mat, SELFIES_ELEMENT_MAP),
        selfies_onehot_mat],
        axis=2)

def selfies_embed_all(selfies_onehot_mat):
    '''
    "Manual embedding" of the 70-element SELFIES onehot vector 
    into 25 elements. Alternative to selfies_encode_mfs

    Parameters
    ----------
    selfies_onehot_mat : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return tf.matmul(selfies_onehot_mat, SELFIES_EMBED_MATRIX)

def selfies_onehot(selfies_mat):
    return tf.one_hot(selfies_mat, len(SELFIES_VOC))

def selfies_split_xy(selfies_mat):
    # if tf.equal(tf.rank(selfies_mat), 2):
    #     return selfies_mat[:,:-1], selfies_mat[:,1:]
    # else:
    return selfies_mat[:,:-1,:], selfies_mat[:,1:,:]

def selfies_decode(selfies_mat, one_hot = True):
    if one_hot:
        selfies_mat = tf.argmax(selfies_mat, axis=2)
    return ts.reduce_join(tf.gather(SELFIES_VOC_TF, selfies_mat), axis=1)

def selfies_crop_sequence(selfies_str,
                  final_char = FINAL_CHAR):
    selfies_split = ts.split(selfies_str, sep=final_char, maxsplit = 1)
    return tf.squeeze(selfies_split.to_tensor(shape=(None,1)))

    

# Interface for decoder:
def ctoy(c):
    return SELFIES_MAP.get(c, 0)

def ytoc(y):
    return SELFIES_VOC[y]

y_tokens = len(SELFIES_VOC)

def sequence_ytoc(seq):
        seq = selfies_decode(seq, one_hot = False) #tf.reshape(seq, [-1, shape[2]]))
        seq = selfies_crop_sequence(seq)
        seq = selfies_to_smiles(seq)
        return seq

@tf.function
def embed_ytox(y):
    return selfies_encode_mf(tf.one_hot(
            tf.expand_dims(y, 1), 
            len(SELFIES_VOC))),

