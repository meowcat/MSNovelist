# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:57:37 2019

@author: stravsmi
"""


INITIAL_CHAR="$"
FINAL_CHAR="*"
PAD_CHAR="&"
SEQUENCE_LEN=128

# Note: the test set does not contain /,\,@ and we can remove those.
# It does contain a small amount of lowercase p!
SMILES_DICT = ['#', '(', ')', '+', '-', '/',  '=', '@','[', '\\', ']', 
              '1', '2', '3', '4', '5', '6','7',
              'c','f','h','i','l','n','o','p','r','s',
              INITIAL_CHAR, FINAL_CHAR, PAD_CHAR]
# SMILES_DICT = [i.encode() for i in SMILES_DICT]


# first row: structural/topological elements
# second row: ring structural/topological elements
# third row: lowercased atom labels
# fourth row: terminators. * for terminal letter, & as zerofill behind
# Note: the PAD_CHAR MUST be last in this implementation, because it will be cut away
# during encoding, so it shouldn't shift the other bits while decoding!

# Element tokens for the formula hinting
ELEMENTS = ['c','f','i','l','n','o','p','r','s']
# Element tokens for the formula input and prediction
ELEMENTS_RDKIT = ['C','F','I','Cl','N','O','P','Br','S','H']

# ELEMENTS = [i.encode() for i in ELEMENTS]
# Note: Hydrogen is removed from the list of elements because it is not 
# implicitely countable in the SMILES string.

# Ring tokens for the grammar hinting.
# Two rules: '[' must match ']', '(' must match ')'
# GRAMMAR_SUMS = [{b'[' : -1, b']' : 1},
#                  {b'(':-1, b')' : 1}]
GRAMMAR_SUMS = [{'[' : -1, ']' : 1},
                 {'(':-1, ')' : 1}]


SMILES_DICT_LEN=len(SMILES_DICT)

SMILES_REPLACE_MAP = {
        b'Br': b'R',
        b'Cl': b'L'
        }

# SMILES_REPLACE_MAP = {
#         'Br': 'R',
#         'Cl': 'L'
#         }


