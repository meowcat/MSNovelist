# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:07:43 2020

@author: stravsm
"""



from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm

import tokens_process as tkp

from . import *

import numpy as np

class ZeroableLayer(Layer):
    '''
    Can be used in two ways:
        By composition:
            zeroable = ZeroableLayer(Dense(44), zero_out = False)
        By subclassing:
            class MyLayer(ZeroableLayer):
                def call(inputs):
                    output = do_things(input)
                    return super().call(output)
    '''
    
    def __init__(self, layer = None, zero_out = False, **kwargs):
            
            super().__init__(**kwargs)
            self.zero_out = zero_out
            
            self.layer = layer
            
            self.zero_out_factor = 1. - (1. * self.zero_out)
            
            self.zero_layer = Lambda(lambda x: self.zero_out_factor * x)
            
    def call(self, inputs, **kwargs):
        
        if self.layer is not None:
            inputs = self.layer(inputs, **kwargs)
        
        return self.zero_layer(inputs)