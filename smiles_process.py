# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:38:12 2018

@author: stravsmi
"""

#from rdkit import Chem
#from rdkit.Chem import AllChem
import numpy as np
import tensorflow
from tensorflow.keras import utils
#from sklearn import preprocessing

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


#smiles = smiles_raw.split('\n')[:-1]
#smiles_tf = [smiles_mapElements(s) for s in smiles]

# Defining the SMILES vocabulary

INITIAL_CHAR="$"
FINAL_CHAR="*"
PAD_CHAR="&"
SEQUENCE_LEN=128

SMILES_DICT = ['#', '(', ')', '+', '-', '/',  '=', '@','[', '\\', ']', 
              '1', '2', '3', '4', '5', '6','7',
              'c','f','h','i','l','n','o','p','r','s',
              INITIAL_CHAR, FINAL_CHAR, PAD_CHAR]
# first row: structural/topological elements
# second row: ring structural/topological elements
# third row: lowercased atom labels
# fourth row: terminators. * for terminal letter, & as zerofill behind

SMILES_DICT_LEN=len(SMILES_DICT)

SMILES_REPLACE_MAP = {
        'Br': 'R',
        'Cl': 'L'
        }



# sklearn mappings are here

class Smiles_ElementMapping(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        return [smiles_mapElements(s) for s in X]

class Smiles_ElementUnmapping(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self
    def transform(self, X):
        return [smiles_unmapElements(s) for s in X]

class Smiles_CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self, Xl):
        X, l = Xl
        Xy = [smiles_encode(s) for s in X]
        X = np.array([el[0] for el in Xy], dtype="bool_")
        y= np.array([el[1] for el in Xy], dtype="bool_")
        return(X,y,l)

class Smiles_CategoricalDecoder(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self, X):
        return np.array([smiles_decode(b) for b in X], dtype="bool_")


class Smiles_Padder(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self, X):
        return np.array([smiles_pad(s) for s in X])


class Smiles_Unpadder(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self, X):
        return[smiles_unpad(s) for s in X]

class Smiles_Slicer(BaseEstimator, TransformerMixin):
    def fit(self,X,y):
        return self
    def transform(self, X):
        Xl = [smiles_slice(s) for s in X]
        X = np.concatenate(Xl)
        l = [len(el) for el in Xl]
        return X, l #[:,0] #, Xt[:,1])



SmilesPreprocessingPipeline = Pipeline(
        memory=None,
        steps=[
                ('map_elements', Smiles_ElementMapping()),
                ('pad', Smiles_Padder()),
                ('slice', Smiles_Slicer()),
                ('encode', Smiles_CategoricalEncoder())                
                ])






# smiles character to corresponding dictionary integer mapping and reverse mapping
smiles_ctoi = dict((c, i) for i, c in enumerate(SMILES_DICT))
smiles_itoc = {v: k for k, v in smiles_ctoi.items()}



# Raw functions are below

def smiles_mapElements(s):
    for k,v in SMILES_REPLACE_MAP.items():
        s = s.replace(k, v)
    return s

def smiles_unmapElements(s):
    for k,v in SMILES_REPLACE_MAP.items():
        s = s.replace(v, k)
    return s

def smiles_encode(s):
    # convert the X part - everything except the last character which is y
    sX = s[:-1]
    # Atom or bond:
    charTypeBits = np.array([(c.islower(), c.isupper()) for c in sX])
    # False,False: structural
    # True, False: aromatic
    # False, True: alipathic
    # then the actual character, lower-cased, as one-hot encoded vector
    charInts = np.array( [smiles_ctoi[c] for c in sX.lower()] )
    charBits = utils.to_categorical(charInts, num_classes=SMILES_DICT_LEN)
    # stack the charTypeBits to the end of the charBits
    X = np.c_[charBits, charTypeBits]
    # convert the last character
    y = s[-1]
    yBits = utils.to_categorical(smiles_ctoy(y), num_classes=2*SMILES_DICT_LEN)    
    return (X, yBits)

def smiles_decode(b):
    [charBits, charTypeBits] = [b[:,:SMILES_DICT_LEN],b[:,SMILES_DICT_LEN:]]
    # map the character one-hot code to the corresponding integer, then to the corresponding character
    charInts = np.array([np.argmax(c) for c in charBits[:,]])
    sLower = np.array([smiles_itoc[i] for i in charInts])
    # convert those characters to uppercase that have the uppercase bit set in the charTypeBits array part
    sCase = np.array([c.upper() if b!=0 else c for c,b in zip(sLower, charTypeBits[:,1])])
    return ''.join(sCase)

def smiles_ctoy(c):
    return smiles_ctoi[c.lower()] + (SMILES_DICT_LEN if c.isupper() else 0)

def smiles_ytoc(i):
    c = smiles_itoc[i % SMILES_DICT_LEN]
    return c if i < SMILES_DICT_LEN else c.upper()

# Pad a smiles code to fixed sequence length
def smiles_pad(s, length=SEQUENCE_LEN,initial = INITIAL_CHAR, final=FINAL_CHAR,pad=PAD_CHAR):
    s = initial + s + final
    s = s.ljust(length, pad)
    return s[:length]

# revert a padded smiles code to original length
def smiles_unpad(s, final=FINAL_CHAR):
    s = s[1:s.find(final)]
    return s

# From one padded smiles, generate [zero to length] training strings for the RNN
def smiles_slice(s, length=SEQUENCE_LEN, pad=PAD_CHAR):
    slen = s.find(pad)+1
    #return [[s[:i].ljust(length, pad) , s[i]] for i in range(slen)]
    return [s[:i].ljust(length, pad) + s[i] for i in range(1,slen)]
    
    
    
    
    