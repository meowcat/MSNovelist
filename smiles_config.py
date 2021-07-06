# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 12:00:18 2018

@author: dvizard
"""

import yaml
import io
import sys
import os
import argparse

parser = argparse.ArgumentParser(
    description='Fingerprint-decoding RNN for de novo compound id.')
parser.add_argument('--config', '-c', nargs="*")

if 'JUPYTER' not in os.environ:
    args = parser.parse_args()
else:
    args = parser.parse_args({})


FILENAME_DEFAULT = "config.yaml"
FILENAME_MACHINE = "config." + os.environ['COMPUTERNAME'] + '.yaml'

config_file = [FILENAME_DEFAULT, FILENAME_MACHINE]
if args.config is not None:
    config_file.extend(args.config)
config = {}


import sys, os

class Error(Exception): pass

def _find(pathname, matchFunc=os.path.isfile):
    for dirname in sys.path:
        candidate = os.path.join(dirname, pathname)
        if matchFunc(candidate):
            return candidate
    raise Error("Can't find file %s" % pathname)

def findFile(pathname):
    return _find(pathname)

def findDir(path):
    return _find(path, matchFunc=os.path.isdir)

def config_reload(config_file = config_file, extra_config = False):
    global config
    if not extra_config:
        config = dict()
    for f in config_file:
        f_ = findFile(f)
        with open(f_, 'r') as stream:
            config_ = yaml.load(stream, Loader=yaml.FullLoader)
            config.update(config_)
    for k, v in config.items():
        if(type(v) == str):
            config[k] = v.replace('.\\', config['base_folder'])
    if "extra_config" in config and not extra_config:
        config_reload(config['extra_config'], extra_config = True)
        
            
            
def config_dump(path):
    with open(path, "w") as stream:
        yaml.dump(config, stream)
    
    
config_reload()

os.environ["HDF5_USE_FILE_LOCKING"]=config['hdf5_lock']