# -*- coding: utf-8 -*-



from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm
from .zeroable_layer import *

class FingerprintFormulaEncoder(tf.keras.layers.Layer):
    
    def __init__(self,
                 layers = [512, 256],
                 layers_decoder = 3,
                 units_decoder = 256,
                 states_per_layer = 2,
                 name = 'encoder',
                 zero_out = False,
                 input_dropout = None,
                 dropout = None,
                 **kwargs):
        
        super().__init__(name = name, **kwargs)        
        self.layers_ = layers
        self.name_ = name
        self.zero_out = zero_out
        self.dropout_ = dropout
        
        self.zeroable = ZeroableLayer(zero_out = self.zero_out)
        
        self.batchnorm = BatchNormalization()
        self.layers = {}
        self.dropout = {}
        for i, units in enumerate(self.layers_):
            self.layers[i] = Dense(units, name=f'{self.name_}_enc_{i}')
            if self.dropout is not None:
                self.dropout[i] = Dropout(self.dropout_[i])
            else:
                self.dropout[i] = None
        self.rnn_starting_states = [
                [Dense(units_decoder, name = f'{self.name_}_states_{i}_{j}', 
                       activation = 'relu') 
                 for j in range(states_per_layer)
                 ]
                for i in range(layers_decoder)
            ]
        
    def call(self, inputs):
        '''
        

        Parameters
        ----------
        inputs : TYPE
            inputs_mf and inputs_fp.

        Returns
        -------
        None.

        '''
        
        layer_stack = concatenate(inputs)
        layer_stack = self.batchnorm(layer_stack)
        for layer, dropout in zip(self.layers.values(), self.dropout.values()):
            layer_stack = layer(layer_stack)
            if dropout is not None:
                layer_stack = dropout(layer_stack, training = True)
        z = layer_stack
        z_ = self.zeroable(z)
        # Transform the z to i*j starting states for the RNN
        rnn_states = [
                [state_layer(z_) for state_layer in state_layers]
                for state_layers in self.rnn_starting_states
            ]
        return z_, rnn_states
        
        
            