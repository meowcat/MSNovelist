# -*- coding: utf-8 -*-

from .blueprinted_model import BlueprintedModel

from tensorflow.keras.layers import *


class DecoderModel(BlueprintedModel):
    def __init__(self,
                 **kwargs):
        
        super().__init__(**kwargs,
                        rnn_unroll = True,
                        rnn_implementation = 2,
                        return_states = True,
                        steps = 1,
                        activation_out = None)
        self.flatten_states_layer = self.FlattenStatesLayer()
        self.unflatten_states_layer = self.UnflattenStatesLayer(self)

            

        
        
    
    def call(self, inputs):
        '''
        

        Parameters
        ----------
        inputs : dictionary with {'tokens_X', 'states', 'counts'}
            DESCRIPTION.

        Returns
        -------
        dict
            a dictionary exactly like the inputs.

        '''
        
        states_flat = inputs['states']
        states = self.unflatten_states_layer(states_flat)
        
        estimated_h_count, mask_vec, hydrogen_estimator_states = self.hydrogen_estimator(
            inputs['tokens_X'],
            initial_state = states['hydrogen_estimator_states'])
        estimated_h_count_ = self.zeroable_hydrogen_estimator(
            estimated_h_count)
                
        auxiliary_counter_input = concatenate([inputs['tokens_X'], 
                                               estimated_h_count_])
        
        auxiliary_counter_input_transformed = self.auxiliary_counter_input_transformer(
            auxiliary_counter_input)
        
        element_grammar_count, auxiliary_counter_states = self.auxiliary_counter(
            auxiliary_counter_input_transformed,
            initial_state = states['auxiliary_counter_states'])
        
        element_grammar_count = self.zeroable_auxiliary_counter(
            element_grammar_count)
        
        z_repeated = self.z_time_transformer(states['z'])
        
        decoder_input = [inputs['tokens_X'],
                         element_grammar_count,
                         z_repeated]
        
        decoder_out, rnn_states = self.sequence_decoder(
            decoder_input, 
            initial_state = states['rnn_states'])
        
        states_out = {
            'auxiliary_counter_states': auxiliary_counter_states,
            'rnn_states': rnn_states,
            'hydrogen_estimator_states': hydrogen_estimator_states,
            'z': states['z']}
        
        states_out_flat = self.flatten_states_layer(states_out)
        
        return {'tokens_X': decoder_out,
                'states': states_out_flat,
                'counts': auxiliary_counter_states
                }
        
        