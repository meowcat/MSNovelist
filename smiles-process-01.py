# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:38:12 2018

@author: stravsmi
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from tensorflow.keras import utils

f = "C:/Cloud/SWITCHdrive/CVAE-molecule/CVAE-master-data/smiles.txt"

d = open(f)
smiles_raw = d.read()
smilesChars = set(smiles_raw)

# We want to transform all two-letter atoms:
# Cl to L
smiles_raw.count('Cl')
smiles_raw.count('C')
smiles_raw.count('l')
# Br to R (note that we have no boron things in the dataset)
smiles_raw.count('Br')
smiles_raw.count('B')
smiles_raw.count('r')
# Fe - there is none


smilesChars_new = set(smiles_mapElements(smiles_raw))

#smiles = smiles_raw.split('\n')[:-1]
#smiles_tf = [smiles_mapElements(s) for s in smiles]

FINAL_CHAR="*"
PAD_CHAR="&"
SEQUENCE_LEN=128

SMILES_DICT = ['#', '(', ')', '+', '-', '/',  '=', '@','[', '\\', ']', 
              '1', '2', '3', '4', '5', '6','7',
              'c','f','h','i','l','n','o','p','r','s',
              FINAL_CHAR, PAD_CHAR]
# first row: structural/topological elements
# second row: ring structural/topological elements
# third row: lowercased atom labels
# fourth row: terminators. * for terminal letter, & as zerofill behind

SMILES_DICT_LEN=len(SMILES_DICT)

# smiles character to corresponding dictionary integer mapping and reverse mapping
smiles_ctoi = dict((c, i) for i, c in enumerate(SMILES_DICT))
smiles_itoc = {v: k for k, v in smiles_ctoi.items()}

def smiles_mapElements(s):
    s = s.replace('Cl', 'L')
    s = s.replace('Br', 'R')
    return s

def smiles_unmapElements(s):
    s = s.replace('L', 'Cl')
    s = s.replace('R', 'Br')
    return s


def smiles_encode(s):
    # Atom or bond:
    charTypeBits = np.array([(c.islower(), c.isupper()) for c in s])
    # False,False: structural
    # True, False: aromatic
    # False, True: alipathic
    # then the actual character, lower-cased, as one-hot encoded vector
    charInts = np.array( [smiles_ctoi[c] for c in s.lower()] )
    charBits = utils.to_categorical(charInts, num_classes=SMILES_DICT_LEN)
    # stack the charTypeBits to the end of the charBits
    b = np.c_[charBits, charTypeBits]
    return b

def smiles_decode(b):
    [charBits, charTypeBits] = [b[:,:SMILES_DICT_LEN],b[:,SMILES_DICT_LEN:]]
    # map the character one-hot code to the corresponding integer, then to the corresponding character
    charInts = np.array([np.argmax(c) for c in charBits[:,]])
    sLower = np.array([smiles_itoc[i] for i in charInts])
    # convert those characters to uppercase that have the uppercase bit set in the charTypeBits array part
    sCase = np.array([c.upper() if b!=0 else c for c,b in zip(sLower, charTypeBits[:,1])])
    return ''.join(sCase)

# Pad a smiles code to fixed sequence length
def smiles_pad(s, length=SMILES_DICT_LEN,final=FINAL_CHAR,pad=PAD_CHAR):
    s = s + final
    s = s.ljust(length, pad)
    return s[:length]

# revert a padded smiles code to original length
def smiles_unpad(s, final=FINAL_CHAR):
    s = s[:s.find(final)]
    return s

smiles = d.read().split('\n')[:-1]

EXPLORE_SIZE = 15

smiles_exp = np.random.choice(smiles, EXPLORE_SIZE)

#smiles_long = smiles_exp.reshape(1)
#smiles_set = set(smiles_exp)

smiles_tf = []
for s in smiles_exp:
    smiles_tf.append(Chem.MolToSmiles(Chem.MolFromSmiles(s),
                     allBondsExplicit=False, isomericSmiles=False))
(smiles_exp, smiles_tf)
