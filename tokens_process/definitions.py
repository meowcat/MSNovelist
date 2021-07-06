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
VOC = ['O',
 '=',
 'C',
 '1',
 'c',
 '2',
 '-',
 '3',
 '4',
 '(',
 ')',
 'n',
 '5',
 '6',
 '[nH]',
 'o',
 'N',
 '[N+]',
 '[O-]',
 'L',
 '[NH+]',
 'S',
 'F',
 's',
 'R',
 '#',
 'P',
 'I',
 '[N-]',
 '7',
 'p',
 '[n+]',
 '[NH3+]',
 '[C-]',
 '[NH2+]',
 '[H]']

VOC.extend([INITIAL_CHAR, FINAL_CHAR, PAD_CHAR])

VOC_TF = tf.convert_to_tensor(np.array(VOC))

element_map_ = [x.upper() for x in VOC]

ELEMENTS = ['C','F','I','L','N','O','P','R','S']
# Element tokens for the formula input and prediction
ELEMENTS_RDKIT = ['C','F','I','Cl','N','O','P','Br','S','H']

elements_vec = lambda x: [e in x for e in ELEMENTS]
ELEMENT_MAP = tf.convert_to_tensor(
    np.array([elements_vec(x) for x in element_map_], dtype='int32'),
    'float32')


GRAMMAR =  [{'(': -1,')': 1}]
GRAMMAR_MAP =tf.convert_to_tensor(
    np.stack(
        [np.sum(np.array([[val * (key in x) for key, val in grammar_.items()] for x in VOC]), axis=1)
         for grammar_ in GRAMMAR], axis=1),
     dtype='float32')


VOC_MAP = {s: i for i, s in enumerate(VOC)}

            
