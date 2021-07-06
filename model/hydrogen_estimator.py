# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:16:39 2020

@author: stravsm
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm


class HydrogenEstimator(Layer):
    
    """
    Hydrogen counting submodule.
    This module is a small forward RNN which takes the X_smiles as input
    and predicts a number which can be interpreted as the contribution of
    hydrogen atoms per SMILES token. It is optimized by summing (averaging)
    over all timesteps and comparing to the real hydrogen count.
    
    The output of the module is then used as an additional hinting tensor
    in analogy to the deterministic atom hinting.
    
    The module operates without knowledge of the fingerprint or the full MF,
    and consequently doesn't use the fingerprint inputs.

    Parameters
    ----------
    X_smiles : TYPE
        DESCRIPTION.
    encoder_states : TYPE, optional
        DESCRIPTION. The default is None.
    return_states : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    def __init__(self, 
                 layers, 
                 units, 
                 name = 'hydrogen_estimator',
                 rnn_implementation = 2,
                 rnn_unroll = False,
                 pad_term_mask = None,
                 return_states = False,
                 **kwargs):
        
        super().__init__(name = name, **kwargs)
        
        self.layers_ = layers
        self.units_ = units
        self.name_ = name
        self.pad_term_mask = pad_term_mask
        self.return_states = return_states
        
        self.layers = {}
        for i in range(self.layers_):
            self.layers[i] = LSTM(
                self.units_,
                return_sequences=True, 
                return_state=True,
                name = f'{self.name_}_rnn_{i}',
                implementation = rnn_implementation, 
                unroll = rnn_unroll)
        
        self.out_layer = TimeDistributed(Dense(1), name = f'{self.name_}_out')
        
        

    def call(self, inputs, initial_state = None):
        
        tokens_input = inputs
        layer_stack = tokens_input
        state_stack = []
        
        if initial_state is None:
            initial_state = [None] * self.layers_
        
        # RNN layers
        for i in range(self.layers_):
            layer_stack_ = self.layers[i](layer_stack, 
                                          initial_state=initial_state[i])
            layer_stack = layer_stack_[0]
            state = layer_stack_[1:]
            state_stack.append(state)
        
        # Final dense layer
        layer_stack = self.out_layer(layer_stack)
        
        # If applicable, mask pad and termination character outputs
        if self.pad_term_mask is not None:
            mask_vec = tf.reduce_max(
                tf.multiply(tokens_input, self.pad_term_mask),
                axis = 2,
                keepdims = True)
            layer_stack = tf.multiply(layer_stack, mask_vec)
            
        if self.return_states:
            return layer_stack, mask_vec, state_stack
        else:
            return layer_stack, mask_vec
        
            
        
    