# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:54:31 2019

@author: stravsmi
"""

from .definitions import *

#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
#import tensorflow
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow import strings as ts
from tensorflow import math as tm
#from sklearn import preprocessing

TF_PAD_CHAR = 38
 

# smiles character to corresponding dictionary integer mapping and reverse mapping
smiles_ctoi = dict((c, i) for i, c in enumerate(SMILES_DICT))
smiles_itoc = {v: k for k, v in smiles_ctoi.items()}

tf_ctoi_ = tf.reshape(
    ts.unicode_decode(SMILES_DICT, 'UTF-8').to_tensor(),
    [-1]).numpy()





def smiles_map_elements(s):
    '''Replace specific SMILES tokens with one-character substitutes
    
    Substitutions are defined in `SMILES_REPLACE_MAP`
    Args:
        s: SMILES to substitute
    Todo:
        Use regex to map 2+ to ++ / 2- to --, and + inside brackets
        to !
    '''
    for k,v in SMILES_REPLACE_MAP.items():
        #s = s.replace(k, v)
        s = ts.regex_replace(s,k,v)
    return s

def smiles_unmap_elements(s):
    '''Replace mapped element characters by original value
    Args:
        s: SMILES to unmap
    '''
    for k,v in SMILES_REPLACE_MAP.items():
        #s = s.replace(v, k)
        s = ts.regex_replace(s,v,k)
    return s

#(experimental_relax_shapes=True)
@tf.function 
def smiles_encode_x(s, mask_char=TF_PAD_CHAR):
    '''Encode string to input
    Note: Characters are not encoded into one-hot vectors, but into an one-hot
        vector for the character and two additional bits for the case.
        All grammar chars are always mapped to lowercase (so some entries are unused).
        (This serves to separate aromaticity information which is determined by
        SMILES case.)
    Args:
        s: SMILES string to encode
        mask_char: For this character, input is zeroed out to enable masking on 0.
            Note that this doesn't work with `CuDNNLSTM`
    Returns:
        np.array, 2-dimensional, axis 0: sequence steps, axis 1: one-hot encoded
        output; second dimension has `SMILES_DICT_LEN + 2` length.
    '''
    if mask_char is not None:
        tf_ctoi_local = tf_ctoi_[tf_ctoi_ != mask_char]
    else:
        tf_ctoi_local = tf_ctoi_
    # Atom or bond:
    s_ = ts.unicode_encode(s, 'UTF-8')
    charTypeBits = tf_chartypebits(s_)
    # False,False: structural
    # True, False: aromatic
    # False, True: alipathic
    # then the actual character, lower-cased, as one-hot encoded vector
    
    # With no masking, the code below has to return 1 for np.sum(charBits, 2)
    # for every SMILES.
    # With masking enabled (default), the sum will be 0 for masked positions.
    charInts = ts.unicode_decode(ts.lower(s_), 'UTF-8').to_tensor()
    charBits = tf.stack([tf.equal(charInts, c) for c in tf_ctoi_local], axis=2)
    # stack the charTypeBits to the end of the charBits
    X = tf.concat([charBits, charTypeBits], axis=2)
    return tf.cast(X, tf.int32)

@tf.function
def smiles_encode_y(s, mask_char=PAD_CHAR):
    '''Encode string to output
    Args:
        s: SMILES string to encode
        mask_char: Unused! Supposed to be used for masking, but isn't.
    Returns:
        np.array, 2-dimensional, axis 0: sequence steps, axis 1: one-hot encoded
        output; second dimension has `SMILES_DICT_LEN * 2` length.
    '''
    # convert the Y sequence: In contrast to the X sequence, this is
    # pure one-hot encoded and doesn't specify bonds etc.
    
    s_ = ts.unicode_encode(s, 'UTF-8')
    charInts = ts.unicode_decode(ts.lower(s_), 'UTF-8').to_tensor()
    charBits = tf.stack([tf.equal(charInts, c) for c in tf_ctoi_], axis=2)
    islower = tf_islower(s_)
    charBitsLower = tf.cast(charBits, tf.int32) * tf.expand_dims(tf.cast(islower, tf.int32), 2)
    charBitsUpper = tf.cast(charBits, tf.int32) * tf.expand_dims(1 - tf.cast(islower, tf.int32), 2)
    charBits = tf.concat([charBitsLower, charBitsUpper], axis=2)
    # Test: 
    # np.array_equal(np.sum(charBits, 2), np.ones((32,128)))
    return charBits

def smiles_ctoy(c):
    '''
    This is probably quite slow and only for use in simple cases right now

    Parameters
    ----------
    c : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    res = ts.unicode_decode([c], 'UTF-8')
    return np.argmax(smiles_encode_y(res))

# helpers to find the upper/lowerness bits with tf.strings functions
def tf_chartypebits(s):
    '''Find chartypebits (i.e. upper, lower or neither) for a string.
    This uses the tf.strings so it can be used in the tf.data pipeline.
    Args:
        s: string in tf.raggedtensor / eagertensor format
    '''
    lower = ts.lower(s)
    upper = ts.upper(s)
    s = ts.unicode_decode(s, 'UTF-8').to_tensor()
    lower = ts.unicode_decode(lower, 'UTF-8').to_tensor()
    upper = ts.unicode_decode(upper, 'UTF-8').to_tensor()
    islower = tm.equal(s, lower)
    isupper = tm.equal(s, upper)
    return tf.stack([isupper, islower], axis=2)

def tf_islower(s):
    lower = ts.lower(s)
    s = ts.unicode_decode(s, 'UTF-8').to_tensor()
    lower = ts.unicode_decode(lower, 'UTF-8').to_tensor()
    islower = tm.equal(s, lower)
    return islower

    
def smiles_ytoc(i):
    '''Decode output to character
    Args: 
        i: argmax of one-hot vector output
        
    Returns:
        the correct character in correct case
    '''
    c = smiles_itoc[i % SMILES_DICT_LEN]
    return c if i < SMILES_DICT_LEN else c.upper()

# Pad a smiles code to fixed sequence length
def smiles_pad(s, length=SEQUENCE_LEN,initial = INITIAL_CHAR, 
               final=FINAL_CHAR,pad=TF_PAD_CHAR,
               pad_direction = "back"):
    '''Pad a SMILES string for LSTM
    
    Pad a SMILES string to fixed length and add start and stop tokens. 
    Args:
        s: SMILES string
        length: total length of the resulting string
        initial: initial token character
        final: final token character
        pad: padding token character
        pad_direction: if "back", the resulting padded SMILES has the original
            SMILES at the beginning, and the pading at the end. "front" is opposite.
    Returns:
        appropriately padded SMILES string
    '''
    s = initial + s + final
    s = ts.unicode_decode(s, 'UTF-8')
    s = s.to_tensor(default_value=pad, shape=(None,length))
    #s = ts.unicode_encode(s, 'UTF-8')
    return s
    # if pad_direction == "back":
    #     s = s.ljust(length, pad)
    #     return s[:length]
    # if pad_direction == "front":
    #     s = s.rjust(length, pad)
    #     return s[-length:]

def smiles_xysplit(s):
    '''Generate offset sequence for LSTM
    
    From an input SMILES (e.g. "$ABCDE*") produce an
    input ("$ABCDE") and output ("ABCDE*") sequence.
    Args:
        s: SMILES string, should be appropriately preprocessed.
    '''
    X = s[:,:-1]
    Y = s[:,1:]
    return X, Y
   
    

def smiles_attach_hints(X_smiles, mol_form = None,
                          formula_hint = True,
                          grammar_hint = True):
    '''
    Attach grammar and formula hints to a SMILES input matrix
    Args:
        X_smiles: np.array, axis 0: sequence steps, axis 1: encoded vocabulary
        mol_form: If given, the predefined molecular formula as an np.array of 
            atom counts per element in the order of `ELEMENTS`. 
        formula_hint: If True, add the formula hint (i.e. a vector with the remaining
            atom counts per element for each sequence step)
        grammar_hint:: If True, add grammar hints (i.e. a vector with the balance
            for each specified token).
    Returns:
        The `X_smiles` input with additional elements on axis 1 as generated by
        the function.
    '''
    # Note about nomenclature:
    # as of now this treats a single training sample (a 2d array)
    # "timesteps" are on axis 0,
    # "smiles chars" (more precisely char and case-bits) are on axis 1
    # "element chars" are the bits on axis 1 which represent atoms (rather than bonds, branches or rings).
    # One X_smiles matrix is a single training sequence here.
    
    # element_bits indexes the element chars in the smiles chars axis. 
    # Extract the submatrix containing only the "element chars" on axis 2.
    if formula_hint:
        element_bits = np.array([smiles_ctoi[el] for el in ELEMENTS])
        X_elements = tf.gather(X_smiles, element_bits, axis=2)
        # get the target formula from the entire string.
        # The formula is obtained by summing the corresponding "element characters" over all "timesteps".
        if mol_form is None:
            mol_form = tm.reduce_sum(X_elements, axis=1)
        X_elsum = tm.cumsum(X_elements, axis=1)
        # Tests:
        # mol_form_final = X_elsum[:,127,:]
        # np.array_equal(mol_form, mol_form_final)
        
        X_elsum = tf.expand_dims(mol_form, 1) - X_elsum
        # Test:
        # np.array_equal(X_elsum[:,127,:], np.zeros((32, 9)))
        # This matrix is attached below X_smiles and returned.
        X_smiles = tf.concat([X_smiles, X_elsum], axis=2)
    
    if grammar_hint:
        X_grammar_rows= tf.concat([grammar_count(X_smiles, grammar_dict) for grammar_dict in GRAMMAR_SUMS], axis=2)
        X_smiles = tf.concat([X_smiles, X_grammar_rows], axis=2)
        
    return tf.cast(X_smiles, tf.int32)


# def initial_formula(X_smiles):
#     '''Return initial formula array
#     Args:
#         X_smiles: np.array with in the format generated by smiles_encode_x.
#     Returns:
#         The molecular formula as an np.array of 
#         atom counts per element in the order of `ELEMENTS`. 
#     '''
#     element_bits = [smiles_ctoi[el] for el in ELEMENTS]
#     X_elements = X_smiles[:,element_bits]
#     return X_elements[:,:].sum(axis=0)



def grammar_count(X_smiles, grammar_dict):
    '''
    Grammar rule to hint closing open brackets etc.
    
    Note that this system now seems useful for binary rules (open ring +1,
    close ring -1) but allows different cases. It is unclear if there is an
    application for this outside of +/- rules. However this means not all
    dictionaries must be of the same size, therefore there is no simple matrix
    operation for all grammars (this function is the operation for a single grammar).
    TODO: fix ring count; ring count is broken in a few exceptional cases where a 2+
    (or possibly 2-) is inside a bracket. (4 in training set)
    The "2-" case occurs frequently because '-' can be a bond. It would be nice to 
    preprocess this out, such that these two '-' get encoded differently.
    Args:
        X_smiles: np.array with in the format generated by smiles_encode_x.
            Axis 0 is sequence steps, axis 1 is input.
        grammar_dict: A list of grammar tokens for one rule, as present in definitions.py
    '''
        # find the tokens relevant to this grammar rule
    X_grammar_tokens = list(grammar_dict.keys())
    # extract the corresponding rows from the X_smiles array and form cumulative sum
    X_grammar_bits = [smiles_ctoi[t] for t in X_grammar_tokens]
    X_grammar_counts = tf.cumsum(
        tf.gather(X_smiles, X_grammar_bits, axis=2), axis=1
        )
    # multiply each token by its factor, and form the sum
    # (e.g. opening bracket is +1, closing bracket is -1)
    X_grammar_factors =  tf.reshape([grammar_dict[t] for t in X_grammar_tokens],
                                    (1,1,-1))
    X_grammar_sum = tf.reduce_sum(X_grammar_counts * X_grammar_factors, 2, keepdims=True)
    # Test:
    # np.array_equal(X_grammar_sum[:,127,:], np.zeros((32,1)))
    return X_grammar_sum


def smiles_decode_prediction(sequence, unmap = True):
    sequence = sequence.argmax(axis=1)
    smiles = ''.join([smiles_ytoc(c) for c in sequence])
    if unmap:
        smiles = smiles_unmap_elements(smiles).numpy().decode('UTF-8')
    return smiles

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

def initial_formula_empty():
    elfo = [0 for el in ELEMENTS]
    return np.array(elfo) 

    


# Interface for decoder:
def ctoy(c):
    return smiles_ctoy(c)

def ytoc(y):
    return smiles_ytoc(y)

y_tokens = SMILES_DICT_LEN * 2
ytoc_array = tf.convert_to_tensor([ytoc(i) for i in range(y_tokens)])

def sequence_ytoc(seq):
        seq = tf.gather(ytoc_array, seq) #tf.reshape(seq, [-1, shape[2]]))
        seq = ts.reduce_join(seq, 1)
        seq = smiles_unmap_elements(seq)
        seq = seq.numpy().astype("str")
        seq = np.array([s.split(FINAL_CHAR)[0] for s in seq])
        return seq

@tf.function
def embed_ytox(y):
    '''
    Embed y output to x input
    
    Embeds the 62 SMILES output tokens into the 2-hot format used as input.

    Parameters
    ----------
    y : Tensor (dtype='int32', shape=(n*k,))
        n*k predictions (chosen top-k * n) for step i

    Returns
    -------
    Tensor (dtype='float32')
        x input for step i+1

    '''
    return tf.expand_dims(tf.gather(embedding_matrix, y, axis=0), 1)


def _init_embedding_matrix():
    '''
    Go through the 2*SMILES_DICT_LEN tokens and find the smiles_x token
    corresponding to the smiles_y token. Even though this is a "two-hot"
    matrix, it works exactly like any regular embedding.

    Returns
    -------
    Tensor of shape (tokens_y, tokens_x).

    '''
    ytox_data = np.array([str.encode(ytoc(i), 'UTF-8') 
                                for i in range(y_tokens)]
                         ).view("uint8").astype("int32")
    ytox_matrix = tf.cast(
        tf.reshape(smiles_encode_x(ytox_data.reshape(1,-1)),
                   (ytox_data.shape[0], -1)), 
        "float32")
    return ytox_matrix
embedding_matrix = _init_embedding_matrix()