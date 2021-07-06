# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:04:16 2020

@author: stravsm
"""


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm

from .recurrent_additive import *

class GlobalSumPooling(tf.keras.layers.Layer):
    
    
    def __init__(self,  name = 'sum_pooling', **kwargs):
        
        super().__init__(name = name, **kwargs)

        
        self.step_counter_factor = Lambda(lambda x: tf.ones_like(x),
                                   name = f'{name}_step_counter_factor')
        self.step_counter = RNN(RecurrentAdditiveCell(units= 1, factor= 1),
                                name=f'{name}_step_counter')
        self.average = GlobalAveragePooling1D(name=f'{name}_average')
    
    def call(self, inputs):
        
        step_counter_factor = self.step_counter_factor(inputs)
        step_counter = self.step_counter(step_counter_factor)
        average = self.average(inputs)
        global_sum = multiply([step_counter, average])
        return global_sum
    
        