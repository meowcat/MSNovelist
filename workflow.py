# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 17:26:18 2018

@author: stravsmi
"""

runfile('smiles_prepare.py')

runfile('smiles_fingerprint.py')

runfile('smiles_transform.py')

c = sc.config
