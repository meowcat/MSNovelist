# -*- coding: utf-8 -*-


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm

class SequenceDecoder(tf.keras.layers.Layer):
    
    def __init__(self,
                 tokens_output,
                 layers = 3,
                 units = 256,
                 name = 'decoder',
                 rnn_implementation = 2,
                 rnn_unroll = False,
                 return_states = False,
                 activation_out = "softmax",
                 **kwargs):
        
        super().__init__(name=name, **kwargs)        
        self.layers_ = layers
        self.name_ = name
        self.units_ = units
        self.tokens_output = tokens_output
        self.return_states = return_states
        self.activation_out = activation_out
        
        self.batchnorm = TimeDistributed(BatchNormalization())
        self.layers = {}
        for i in range(self.layers_):
            self.layers[i] = LSTM(self.units_, 
                                  name=f'{self.name_}_rnn_{i}',
                                  return_sequences = True,
                                  return_state = True,
                                  implementation = rnn_implementation, 
                                  unroll = rnn_unroll)
        self.out_layer = Dense(self.tokens_output, 
                               activation = self.activation_out,
                               name = f'{self.name_}_out')

    def call(self, inputs, initial_state = None):
        
        
        if initial_state is None:
            initial_state = [None] * self.layers_
        
        layer_stack = concatenate(inputs)
        layer_stack = self.batchnorm(layer_stack)
        state_stack = []
        
        # RNN layers
        for i in range(self.layers_):
            layer_stack_ = self.layers[i](layer_stack, 
                                          initial_state=initial_state[i])
            layer_stack = layer_stack_[0]
            state = layer_stack_[1:]
            state_stack.append(state)
        
        layer_stack = self.out_layer(layer_stack)

        if self.return_states:
            return layer_stack, state_stack
        else:
            return layer_stack
        

