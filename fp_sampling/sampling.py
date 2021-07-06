# -*- coding: utf-8 -*-
"""

@author: dvizard
"""


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import math as tm
from tensorflow import strings as ts
from tensorflow import data as td

from matplotlib import pyplot as plt



class Sampler:
    def __init__(self):
        pass
    
    def map_dataset(self, dataset):
        fp = dataset["fingerprint"]
        fp_sampled = fp.map(self.sample)
        dataset.update({"fingerprint_sampled": fp_sampled})
        return dataset
    
    @tf.function
    def sample(self, fp):
        raise NotImplementedError("Abstract method 'sample' not implemented in Sampler")
        
    def demo(self, evaluation_data):
        fingerprints_true = evaluation_data["fingerprints"]
        fingerprints_predicted = evaluation_data["fingerprints_degraded"]
        
        pass
    
        
