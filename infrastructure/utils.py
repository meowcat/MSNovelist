# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:11:48 2020

@author: stravsm
"""

import os
import pandas as pd

class EvaluationLogger:
    
    def __init__(self, path_base, config, eval_id, eval_counter, pickle_id):
        
        self.path_base = path_base
        self.config = config
        self.eval_id = eval_id
        self.eval_counter = eval_counter
        self.pickle_id = pickle_id
    
    def csvpath(self, key):
        csvpath_ = os.path.join(
            self.config["eval_folder"],
            f"eval_{self.eval_id}_{self.path_base}_{key}.csv")
        return csvpath_
    
    def logpath(self, key):
        logpath_ = os.path.join(
            self.config["eval_folder"],
            f"eval_{self.eval_id}_{self.path_base}_{key}.txt")
        return logpath_        
        
    def append_csv(self, key, data, remove_cols = ['mol_ref', 'fingerprint_ref_true', 'fingerprint_ref']):
        path = self.csvpath(key)
        data_store = data.copy()
        data_store.reset_index(inplace=True)
        data_store.insert(0, "eval_id", self.eval_id)
        data_store.insert(0, "eval_counter", self.eval_counter)
        data_store.insert(0, "pickle_id", self.pickle_id)
        data_store.insert(0, "weights", self.config["weights"])
        data_store.insert(0, "model_tag", self.config["model_tag"])
        data_store.insert(0, "sampler_name", self.config["sampler_name"])
        
        for col in remove_cols:
            if col in data_store.columns:
                del data_store[col]

        try:
            data_append = pd.read_csv(path)
            data_append = data_append.append(data_store)
        except FileNotFoundError:
            data_append = data_store
        data_append.to_csv(path, index=False)
    
    # https://stackoverflow.com/a/51593236/1259675
    def print_full(self, x, **kwargs):
        pd.set_option('display.max_rows', len(x))
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        #pd.set_option('display.float_format', '{:20,.2f}'.format)
        #pd.set_option('display.max_colwidth', None)
        print(x, **kwargs)
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.float_format')
        pd.reset_option('display.max_colwidth')
    
    def append_txt(self, key, data = {}, **kwargs):
        with open(self.logpath(key), 'a+') as logfile:
            print("Evaluation ID:", self.eval_id, file = logfile)
            print("Evaluation counter:", self.eval_counter, file = logfile)
            print("Pickle ID:", self.pickle_id, file = logfile)
            print("Weights:", self.config['weights'], file = logfile)
            print("Label:", self.config['model_tag'], file = logfile)
            print("Sampler:", self.config['sampler_name'], file = logfile)
            for ekey, element in data.items():
                if isinstance(element, pd.DataFrame):
                    print(f"{ekey}:", file = logfile)
                    self.print_full(element, **kwargs, file = logfile)
                    print("", file = logfile)
                elif isinstance(element, pd.Series):
                    print(f"{ekey}:", file = logfile)
                    print(element, **kwargs, file = logfile)
                    print("", file = logfile)
                else:
                    print(f"{ekey}:", element, **kwargs, file=logfile)
                
                
            print("=================================================", file=logfile)


