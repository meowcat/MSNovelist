# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:00:18 2018

@author: dvizard
"""

import yaml
import io
import sys
import os



FILENAME_DEFAULT = "config.yaml"

config_file = FILENAME_DEFAULT
if len(sys.argv) > 1:
    config_file = sys.argv[1]
config = {}

def config_reload(f = config_file):
    global config
    with open(config_file, 'r') as stream:
        config = yaml.load(stream)
    for k, v in config.items():
        if(type(v) == str):
            config[k] = v.replace('.\\', config['base_folder'])
            
config_reload()

os.environ["HDF5_USE_FILE_LOCKING"]=config['hdf5_lock']