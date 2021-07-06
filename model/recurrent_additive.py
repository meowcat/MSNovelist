# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:03:47 2020

@author: stravsm
"""


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm


class RecurrentAdditiveCell(tf.keras.layers.Layer):
    def __init__(self, units, factor, **kwargs):

        super().__init__(**kwargs)
        
        self.units = units
        self.state_size = units
        self.factor = factor

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        output = prev_output + self.factor * inputs
        return output, [output]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({'factor': self.factor})
        return config