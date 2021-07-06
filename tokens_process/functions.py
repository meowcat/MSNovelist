# -*- coding: utf-8 -*-
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



def tokens_to_tf_wrap(smiles):
    tokens_ = tokens_from_smiles(ts_decode(smiles))
    tokens_ = tokens_pad(tokens_)
    tokens_mat = tokens_encode(tokens_)
    return tokens_mat

def tokens_to_tf(smiles):
    
    tokens_mat = tf.py_function(
        tokens_to_tf_wrap,
        [smiles],
        Tout = 'int32')
    tokens_mat.set_shape(tf.TensorShape([None, None]))
    return tokens_mat


def tokens_from_smiles(smiles):
    tokens = [re.findall('[^[]|\[.*?\]', s.replace('Cl','L').replace('Br','R')) for s in smiles]
    return tokens
    
def tokens_to_smiles(tokens):
    return [bytes.decode(x).replace('L','Cl').replace('R','Br') for x in tokens.numpy()]
    

def tokens_pad_one(tokens, 
                length = SEQUENCE_LEN, 
                initial_char = INITIAL_CHAR, 
                final_char = FINAL_CHAR,
                pad_char = PAD_CHAR):
    tokens.insert(0, initial_char)
    tokens.append(final_char)
    tokens.extend([pad_char] * (length - len(tokens)))
    return tokens[:length]
    
def tokens_pad(tokens, 
                length = SEQUENCE_LEN, 
                initial_char = INITIAL_CHAR, 
                final_char = FINAL_CHAR,
                pad_char = PAD_CHAR):
    return [tokens_pad_one(x, length, initial_char, final_char, pad_char) for x in tokens]
    

def tokens_encode_one(tokens):
    return [VOC_MAP.get(c, 0) for c in tokens]

def tokens_encode(tokens):
    return tf.stack([tokens_encode_one(x) for x in tokens], axis=0)

def tokens_encode_mf(tokens_onehot_mat):
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
        tf.matmul(tokens_onehot_mat, ELEMENT_MAP),
        tokens_onehot_mat],
        axis=2)

def tokens_embed_all(tokens_onehot_mat):
    '''
    Adding MF and grammar

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
            tf.matmul(tokens_onehot_mat, ELEMENT_MAP),
            tf.matmul(tokens_onehot_mat, GRAMMAR_MAP),
            tokens_onehot_mat],
            axis=2)

def tokens_augment(tokens_onehot_mat):
    return tf.concat([
        tf.matmul(tokens_onehot_mat, ELEMENT_MAP),
        tf.matmul(tokens_onehot_mat, GRAMMAR_MAP),
        ],
        axis=2)

def tokens_onehot(tokens_mat):
    return tf.one_hot(tokens_mat, len(VOC))

def tokens_split_xy(tokens_mat):
    # if tf.equal(tf.rank(selfies_mat), 2):
    #     return selfies_mat[:,:-1], selfies_mat[:,1:]
    # else:
    return tokens_mat[:,:-1,:], tokens_mat[:,1:,:]

def tokens_decode(tokens_mat, one_hot = True):
    if one_hot:
        tokens_mat = tf.argmax(tokens_mat, axis=2)
    return ts.reduce_join(tf.gather(VOC_TF, tokens_mat), axis=1)

def tokens_crop_sequence(tokens_str,
                  final_char = FINAL_CHAR):
    tokens_split = ts.split(tokens_str, sep=final_char, maxsplit = 1)
    return tf.squeeze(tokens_split.to_tensor(shape=(None,1)))

    

# Interface for decoder:
def ctoy(c):
    return VOC_MAP.get(c, PAD_CHAR)

def ytoc(y):
    return VOC[y]

y_tokens = len(VOC)

def sequence_ytoc(seq):
        seq = tokens_decode(seq, one_hot = False) #tf.reshape(seq, [-1, shape[2]]))
        seq = tokens_crop_sequence(seq)
        seq = tokens_to_smiles(seq)
        return seq

@tf.function
def embed_ytox(y):
    return tf.one_hot(
            tf.expand_dims(y, 1), 
            len(VOC))

