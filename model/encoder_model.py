# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:08:16 2020

@author: stravsm
"""


import tensorflow as tf

from .blueprinted_model import BlueprintedModel

from tensorflow.keras.layers import Lambda

class EncoderModel(BlueprintedModel):
    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.hydrogen_estimator_one_state_creator = Lambda(
            lambda x: tf.zeros(
                (tf.shape(x)[0],
                 tf.convert_to_tensor(self.config['hcount_hidden_size']))
                ))
        
        self.hydrogen_estimator_states_creator = Lambda(
            lambda x: [[x, x]] * self.config['hcounter_layers']
            )
        
        self.initial_token_creator = Lambda(
            lambda x: tf.repeat(self.initial_char,
                                tf.shape(x)[0],
                                axis=0)
            )
        
        self.flatten_states_layer = self.FlattenStatesLayer()
        
    def call(self, inputs):
        '''


        Parameters
        ----------
        inputs: 
            dictionary with {'fingerprint', 'mol_form'}

        Returns
        -------
            dictionary with {'auxiliary_counter_states', 
                            'rnn_states', 
                            'hydrogen_estimator_states', 
                            'z', 
                            'tokens_y'}

        '''
        
        fingerprints = inputs['fingerprint_selected']
        fingerprints_ = self.fingerprint_rounding(fingerprints)        
        
        z, rnn_states = self.encoder([fingerprints_,
                                                  inputs['mol_form']])
        
        hydrogen_estimator_one_state = self.hydrogen_estimator_one_state_creator(
            inputs['mol_form'])
        
        hydrogen_estimator_states = self.hydrogen_estimator_states_creator(
            hydrogen_estimator_one_state)        
        
        auxiliary_counter_states = self.auxiliary_counter_start_state_transformer(
            inputs['mol_form'])
        
        initial_tokens = self.initial_token_creator(inputs['mol_form'])
        
        states_out = {
            'auxiliary_counter_states': auxiliary_counter_states,
            'rnn_states': rnn_states,
            'hydrogen_estimator_states': hydrogen_estimator_states,
            'z': z}
        
        states_flat = self.flatten_states_layer(states_out)
        
        return {
            'tokens_X': initial_tokens,
            'states': states_flat #,
            #'counts': auxiliary_counter_states
            }
        
        
        