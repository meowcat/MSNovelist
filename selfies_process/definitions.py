# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:24:57 2020

@author: stravsm
"""


INITIAL_CHAR="$"
FINAL_CHAR="*"
PAD_CHAR="&"
SEQUENCE_LEN=128


import selfies as sf
import numpy as np
import tensorflow as tf
import re
import itertools

# The dictionary below was obtained from the cococoh dataset
# converted to selfies, then sf.get_alphabet_from_selfies

SELFIES_VOC = [
 #'[unk]',
 '[#C-expl]',
 '[#C]',
 '[#N+expl]',
 '[#N]',
 '[#P]',
 '[#S+expl]',
 '[#S]',
 '[=C-expl]',
 '[=CH-expl]',
 '[=C]',
 '[=Cl]',
 '[=I]',
 '[=N+expl]',
 '[=N-expl]',
 '[=NH+expl]',
 '[=N]',
 '[=O+expl]',
 '[=OH+expl]',
 '[=O]',
 '[=P+expl]',
 '[=P]',
 '[=S]',
 '[Br]',
 '[Branch1_1]',
 '[Branch1_2]',
 '[Branch1_3]',
 '[Branch2_1]',
 '[Branch2_2]',
 '[Branch2_3]',
 '[C+expl]',
 '[C-expl]',
 '[CH+expl]',
 '[CH-expl]',
 '[CH2+expl]',
 '[CH2-expl]',
 '[C]',
 '[Cl]',
 '[Expl#Ring1]',
 '[Expl#Ring2]',
 '[Expl=Ring1]',
 '[Expl=Ring2]',
 '[F]',
 '[Hexpl]',
 '[I+expl]',
 '[I]',
 '[N+expl]',
 '[N-expl]',
 '[NH+expl]',
 '[NH-expl]',
 '[NH2+expl]',
 '[NH3+expl]',
 '[NHexpl]',
 '[N]',
 '[O+expl]',
 '[O-expl]',
 '[OH2+expl]',
 '[O]',
 '[P+expl]',
 '[PH+expl]',
 '[PHexpl]',
 '[P]',
 '[Ring1]',
 '[Ring2]',
 '[S+expl]',
 '[S-expl]',
 '[SH+expl]',
 '[S]']
SELFIES_VOC.extend([INITIAL_CHAR, FINAL_CHAR, PAD_CHAR])

SELFIES_VOC_TF = tf.convert_to_tensor(np.array(SELFIES_VOC))

selfies_element_map_ = [sf.decoder(x).replace('Cl','L').replace('Br','R') for x in SELFIES_VOC]

ELEMENTS = ['C','F','I','L','N','O','P','R','S']
# Element tokens for the formula input and prediction
ELEMENTS_RDKIT = ['C','F','I','Cl','N','O','P','Br','S','H']

elements_vec = lambda x: [e in x for e in ELEMENTS]
SELFIES_ELEMENT_MAP = tf.convert_to_tensor(
    np.array([elements_vec(x) for x in selfies_element_map_], dtype='int32'),
    'float32')

BONDTYPES = ['=', '#']
bondtypes_vec = lambda x: [t in x for t in BONDTYPES]
selfies_bondtypes_map_ = tf.convert_to_tensor(
    np.array([bondtypes_vec(x) for x in SELFIES_VOC], dtype='float32'),
    )
SELFIES_BONDTYPES_MAP = tf.concat([
    selfies_bondtypes_map_,
    1.-tf.reduce_sum(selfies_bondtypes_map_, axis=1, keepdims=True)
    ], axis=1)

#SELFIES_EXPL_ITEMS = set(itertools.chain(*[re.findall('H?[0-9]?[+-]?expl', s) for s in SELFIES_VOC]))
SELFIES_H_ITEMS = set(itertools.chain(*[re.findall('H[0-9]?', s) for s in SELFIES_VOC]))
SELFIES_CHARGE_ITEMS = set(itertools.chain(*[re.findall('[+-]expl', s) for s in SELFIES_VOC]))
SELFIES_STRUCT_ITEMS = set(itertools.chain(*[re.findall('Ring[0-9]|Branch[0-9]_[0-9]', s) for s in SELFIES_VOC]))
SELFIES_EXTRA_ITEMS = list(SELFIES_H_ITEMS) + list(SELFIES_CHARGE_ITEMS) + list(SELFIES_STRUCT_ITEMS)

items_vec = lambda x: [item in x for item in SELFIES_EXTRA_ITEMS]
SELFIES_ITEMS_MAP = tf.convert_to_tensor(
    np.array([items_vec(x) for x in SELFIES_VOC], dtype='float32'),
    )
SELFIES_EMBED_MATRIX = tf.concat([
    SELFIES_ELEMENT_MAP,
    SELFIES_BONDTYPES_MAP,
    SELFIES_ITEMS_MAP
    ], axis=1)

SELFIES_MAP = {s: i for i, s in enumerate(SELFIES_VOC)}
SELFIES_UNMAP = {i: s for i, s in enumerate(SELFIES_VOC)}

            
