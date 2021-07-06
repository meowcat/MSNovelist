# -*- coding: utf-8 -*-


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm

from . import *

import tokens_process as tkp

class TranscoderModel(BlueprintedModel):
    '''
    This is the transcoder model used for training
    '''
    
    
    def compile(self, **kwargs):
        
        compile_args = {
            'optimizer': 'Adam',
            # 'loss': {
            #     'tokens_y': 'categorical_crossentropy', 
            #      'n_hydrogen': 'mse'},
            # 'loss_weights': {
            #     'tokens_y': 1,
            #     'n_hydrogen': 0.01}
            'loss': ['categorical_crossentropy', 'mse'],
            'loss_weights': [1, 0.01]
            }
        
        compile_args.update(kwargs)
        
        super().compile(
            **compile_args)
    
    
    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs,
                         rnn_unroll = False)
        
        
    
    def call(self, inputs):
        '''
        

        Parameters
        ----------
        inputs : dictionary with {'mol_form', 'fingerprint', 'tokens_X'}
            DESCRIPTION.

        Returns
        -------
        list
            [smiles_Y, n_hydrogen]

        '''
        
        formula = inputs['mol_form']
        
        fingerprints = inputs['fingerprint_selected']
        fingerprints_ = self.fingerprint_rounding(fingerprints)        
        
        z, decoder_initial_states = self.encoder([fingerprints_,
                                                  inputs['mol_form']])
        
        estimated_h_count, _ = self.hydrogen_estimator(inputs['tokens_X'])
        
        estimated_h_sum = self.hydrogen_sum(estimated_h_count)
        
        estimated_h_count_ = self.zeroable_hydrogen_estimator(
            estimated_h_count)
        
        estimated_h_count_ = self.hcounter_gradient_stop(estimated_h_count_)
        
        auxiliary_counter_input = concatenate([inputs['tokens_X'], 
                                               estimated_h_count_])
        
        auxiliary_counter_start_states = self.auxiliary_counter_start_state_transformer(
            inputs['mol_form'])
        
        auxiliary_counter_input_transformed = self.auxiliary_counter_input_transformer(
            auxiliary_counter_input)
        
        element_grammar_count = self.auxiliary_counter(
            auxiliary_counter_input_transformed,
            initial_state = auxiliary_counter_start_states)
        
        element_grammar_count = self.zeroable_auxiliary_counter(
            element_grammar_count)
        
        z_repeated = self.z_time_transformer(z)
        
        decoder_input = [inputs['tokens_X'],
                         element_grammar_count,
                         z_repeated]
        
        decoder_out = self.sequence_decoder(decoder_input, decoder_initial_states)
        
        # self.output_names = ['tokens_y', 'n_hydrogen']
        
        # return {'tokens_y': decoder_out, 
        #         'n_hydrogen': estimated_h_sum}
        return decoder_out, estimated_h_sum #, element_grammar_count
        
        
        
        
        
        
        