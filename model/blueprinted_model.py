

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
from tensorflow import math as tm

import tokens_process as tkp

import logging
logger = logging.getLogger("MSNovelist")

from . import *

import numpy as np


class BlueprintedModel(Model):
    
    
    class FlattenStatesLayer(Layer):
        '''
        Go from the states in the model format (z, vi, SG, SH) to a single vector.
        '''
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        
        def call(self, inputs):
            '''
            inputs : dictionary with {'auxiliary_counter_states', 
                            'rnn_states', 
                            'hydrogen_estimator_states', 
                            'z', 
                            'tokens_X'}
            '''
            rnn_states = inputs['rnn_states']
            rnn_states = [tf.stack(x, axis=1) for x in rnn_states]
            rnn_states = tf.stack(rnn_states, axis=1)
            rnn_states = tf.reshape(rnn_states, (tf.shape(rnn_states)[0], -1))
            
            hydrogen_estimator_states = inputs['hydrogen_estimator_states']
            hydrogen_estimator_states = [tf.stack(x, axis=1) 
                                         for x in hydrogen_estimator_states]
            hydrogen_estimator_states = tf.stack(hydrogen_estimator_states, axis=1)
            hydrogen_estimator_states = tf.reshape(hydrogen_estimator_states, 
                                                   (tf.shape(hydrogen_estimator_states)[0], -1))
            
            states = tf.concat(
                [inputs['auxiliary_counter_states'],
                hydrogen_estimator_states,
                rnn_states,
                inputs['z']],
                axis=1
                )
            return states
            
    class UnflattenStatesLayer(Layer):
        def __init__(self, model, **kwargs):
            
            super().__init__(**kwargs)
            self.states = {
                'auxiliary_counter_states': (model.auxiliary_counter_units, ),
                'hydrogen_estimator_states': (
                    model.config['hcounter_layers'],
                    2,
                    model.config['hcount_hidden_size']
                    ),
                'rnn_states': (
                    model.config['decoder_layers'],
                    2,
                    model.config['decoder_hidden_size']
                    ),
                'z': (model.config['fp_enc_layers'][-1],)
                }
            self.states_shapes = [(-1,) + shape for shape in self.states.values()]
            self.states_length = [np.prod(spec) for spec in self.states.values()]
            self.states_pos = np.concatenate([[0], np.cumsum(self.states_length[:-1])])
            
        def call(self, inputs):
             states_split = [
                 inputs[:,start:start+length]
                 for start, length in zip(self.states_pos, self.states_length)
                 ]
             
             # Reshapes every statevector to the appropriate tensor
             # (i.e. 2*2*32 states get reshaped into a (-1, 2,2,32) tensor
             # with -1 being the batch dimension)
             states_reshape = [tf.reshape(state, shape)
                               for state, shape 
                               in zip(states_split, self.states_shapes)]
             
             # Now just unstack by hand, or I go insane
             hydrogen_estimator_states = states_reshape[1]
             hydrogen_estimator_states = tf.unstack(hydrogen_estimator_states, axis=1)
             hydrogen_estimator_states = [tf.unstack(x, axis=1) for x in hydrogen_estimator_states]
             
             rnn_states = states_reshape[2]
             rnn_states = tf.unstack(rnn_states, axis=1)
             rnn_states = [tf.unstack(x, axis=1) for x in rnn_states]
             
             return {
                 'auxiliary_counter_states': states_reshape[0],
                 'hydrogen_estimator_states': hydrogen_estimator_states,
                 'rnn_states': rnn_states,
                 'z': states_reshape[3]
                 }
    
    
    def copy_weights(self, model):
        
        layers_ref = [l.name for l in model.layers]
        
        for layer in self.layers:
            if layer.built and (layer.name in layers_ref):
                logger.info(f"Loading layer {layer.name} weights")
                try:
                    layer_ = model.get_layer(layer.name)
                    layer.set_weights(layer_.get_weights())
                    logger.info("Loaded")
                except:
                    logger.info("Error, probably no such layer")
        
    
    def construct_counter_matrix(self):
        m11 = tkp.ELEMENT_MAP
        m13 = tkp.GRAMMAR_MAP
        m12 = tf.zeros_like(m13)
        mleft = tf.concat([m11, m12, m13], axis=1)
        m21 = tf.zeros_like(m11[:1,:])
        m22 = tf.ones_like(m12[:1,:])
        m23 = tf.zeros_like(m13[:1,:])
        mright = tf.concat([m21, m22, m23], axis=1)
        return tf.concat([mleft, mright], axis=0)
    
    
    def __init__(self,
                 blueprints,
                 config = {},
                 return_states = False,
                 rnn_unroll = False,
                 rnn_implementation = 2,
                 steps = None,
                 round_fingerprints = False,
                 activation_out = "softmax",
                 **kwargs):
        
        super().__init__(**kwargs)
        
        # Set default config and update from supplied config
        self.config = {
            'decoder_hidden_size': 256,
            'hcount_hidden_size': 32,
            'fp_enc_layers': [512, 256],
            'loss_weights': 
                {'out_smiles': 1,
                'out_nhydrogen': 0.03},
            'hcounter_layers': 2,
            'decoder_layers': 3,
            'use_hydrogen_estimator': True,
            'use_auxiliary_counter': True,
            'use_fingerprint': True
                }
        if "model_config" in config:
            config_ = config["model_config"]
            self.config.update(config_)
            
        self.round_fingerprints = round_fingerprints

        # Fixed values: blueprints and losses,
        # rules for hinting (element bits)
        
        self.blueprints = blueprints
        self.shapes = {k: var.shape[1:] for k, var in blueprints.items()}
        
        # # (Heavy) elements are on top (0..8), in both the augmented and the embedded matrix.
        # self.element_bits = np.arange(len(tkp.ELEMENTS))
        # self.grammar_bits = max(self.element_bits) + 1 + np.arange(len(tkp.GRAMMAR))
        # self.token_bit_count = self.blueprints["tokens_y"].shape[2]
        # self.total_bit_count = self.blueprints["tokens_X"].shape[2]
        # self.token_bits = np.arange(self.total_bit_count - self.token_bit_count,
        #                             self.total_bit_count)
        
        self.initial_char = (
            tf.expand_dims(tkp.tokens_onehot(
                tkp.tokens_encode_one(
                    tkp.INITIAL_CHAR)),0
                ))
        self.pad_mask = (tf.expand_dims(tkp.tokens_onehot(tkp.tokens_encode_one(tkp.PAD_CHAR)), 0))
        self.term_mask = (tf.expand_dims(tkp.tokens_onehot(tkp.tokens_encode_one(tkp.FINAL_CHAR)), 0))
        self.hcount_mask = tf.ones_like(self.pad_mask) - self.pad_mask - self.term_mask
        
        self.counter_matrix = self.construct_counter_matrix()
        
        
        
        self.tokens_output = self.shapes['tokens_y'][1]
        self.auxiliary_counter_units = self.counter_matrix.shape[1]
        self.steps = steps or self.shapes['tokens_y'][0]
    
    
    
        self.encoder = FingerprintFormulaEncoder(
            layers = self.config['fp_enc_layers'],
            layers_decoder = self.config['decoder_layers'],
            units_decoder = self.config['decoder_hidden_size'],
            zero_out = not self.config['use_fingerprint']
            )
        
        self.hydrogen_estimator = HydrogenEstimator(
            layers = self.config['hcounter_layers'], 
            units = self.config['hcount_hidden_size'],
            pad_term_mask = self.hcount_mask,
            rnn_unroll = rnn_unroll,
            rnn_implementation = rnn_implementation,
            return_states = return_states
            )
        # This must be separate because we still need to train the H layer,
        # even if we don't use it, because otherwise this will still show
        # up in the loss.
        self.zeroable_hydrogen_estimator = ZeroableLayer(
            zero_out = not self.config['use_hydrogen_estimator'])
        
        self.hydrogen_sum = GlobalSumPooling(name = 'n_hydrogen')
        
        self.auxiliary_counter_input_transformer = Lambda(
            lambda x: tf.matmul(x, self.counter_matrix))
        
        self.auxiliary_counter_start_state_transformer = Lambda(
            lambda x: tf.pad(x, [[0,0], [0,1]]))
        
        self.auxiliary_counter = RNN(RecurrentAdditiveCell(
                    units = self.auxiliary_counter_units,
                    factor = -1
                ),
                return_sequences = True,
                return_state = return_states,
                unroll = rnn_unroll
                )
        
        self.zeroable_auxiliary_counter = ZeroableLayer(
            zero_out = not self.config['use_auxiliary_counter'])
        
        self.sequence_decoder = SequenceDecoder(
            tokens_output = self.tokens_output,
            layers = self.config['decoder_layers'],
            units = self.config['decoder_hidden_size'],
            rnn_unroll = rnn_unroll,
            rnn_implementation = rnn_implementation,
            return_states = return_states,
            name = 'tokens_y',
            activation_out = activation_out
            )
        
        self.z_time_transformer = RepeatVector(self.steps)
        
        self.hcounter_gradient_stop = Lambda(lambda x: tf.stop_gradient(x),
                                  name = 'hcounter_gradient_stop')
        
        if self.round_fingerprints:
            self.fingerprint_rounding = Lambda(lambda x: tf.round(x))
        else:
            self.fingerprint_rounding = Lambda(lambda x: x)
        
        
