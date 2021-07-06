# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 09:53:53 2020

@author: stravsm
"""



import smiles_process as sp
from tensorflow import strings as ts
from tensorflow import math as tm
import tensorflow as tf
import numpy as np
import pandas as pd
import importlib

from .decoder_base import DecoderBase

class StochasticSampler(DecoderBase):
    
    def __init__(self, encoder, decoder, steps, n, k, kk, temperature = 1., config = {},
                 generator = tf.random.experimental.get_global_generator()):
        
        
        super().__init__(
            encoder, decoder, steps, n, k, kk, temperature, config = config)        
        self.generator = generator
    

    def _init_templates(self):
        '''
        Generate the constant inputs used at step 0 of decoding.
        
        Note: for each of n sequences, only one of the inputs is valid and
        starts with y = initial_char, score = 0, the others start with
        score = -Inf! The "valid" sequence is put at k-1 rather than 0
        to facilitate debugging regarding the reshape, intdiv and modulo
        operations.        

        Returns
        -------
        y_init : numpy.array (n*k,)
            A flattened array with the initial_char as prediction for one
            sequence in the beam, which is fed to the embedding matrix 
            to generate the first x input.
        scores_init : np.array(n*k,)
            A flattened array with score zero (for the starting sequence)
            or -Inf (for all empty positions)
        pad_mask : np.array(n*k, 1, y_tokens)
            An array in the shape of the model y output
            that adds a score of -Inf for every y result of pad_char.

        '''
        y_init = np.ones((self.n,self.k), dtype='int32') * self.initial_char
        y_init = tf.convert_to_tensor(np.reshape(y_init, (-1,)))
        # y_init = tf.convert_to_tensor(y_init, dtype="int32")
        # All invalid inputs start with a score of -infinity so they are only
        # ever continued if there is not enough valid possibilities (i.e. in the
        # first step when k > tokens)
        scores_init = np.full((self.n,self.k), 0, dtype='float32')
        scores_init = tf.convert_to_tensor(np.reshape(scores_init, (-1,)))
        # Define a padding mask that pushes the score of PAD_CHAR down to
        # minus infinity, such that these sequences do not suppress
        # close competitors. This will lead to a lot of additional long crap
        # being generated, but we don't care as we only backtrace a select 
        # number of good sequences.
        pad_mask = np.zeros((1, 1, self.y_tokens), dtype='float32')
        pad_mask[0,0,self.pad_char] = -np.inf
        pad_mask = tf.convert_to_tensor(
            np.reshape(pad_mask, (1,1,-1)))
        # pad_mask[:,:,self.pad_char] = -np.inf
        # pad_mask = np.reshape(pad_mask, (self.n*self.k,1,-1))
        # Return:
        return y_init, scores_init, pad_mask
    
    
    @tf.function
    def beam_step(self, y, states, scores, counts):
        '''
        Performs the beam search priorization for one decoding step.
        In (for now for a single query sequence) the k (beam width) states and
        results from step n, as well as the cumulated, non-penalized scores for
        every of k "parents". Returns the k next query sequences and states
    
        Parameters
        ----------
        y : Tensor k x 1 x y_tokens
            The prediction results for k beam candidates from step n prediction
        states : Tensor k x tensor_size
            The predicted states of k beam candidates after step n prediction
        scores : Tensor k x 1
            Cumulated scores of candidates 0..k-1 at step n-1, going into step
            n prediction
        counts: Tensor k x 1 x (element_tokens + grammar_tokens)
            Remaining element count. Kill the sequence if any goes <0.
        
        Returns
        -------
        Tuple (ymax, ysequence, states, scores)
        ymax : Tensor k x 1
            The chosen characters for k beam candidates after step n priorization
        ysequence : Tensor k x 1
            The chosen parent sequences for k beam candidates after 
            step n priorization; to be converted ("embedded") into the input 
            characters for step n+1 prediction
        states : Tensor k x 1
            The chosen k candidate states after step n priorization
            (the input states for step n+1 prediction)
        scores : Tensor k x 1
            The cumulated scores after step n priorization
            (the input scores for step n+1 priorization)    
        '''
        # Make a y-shaped "kill invalid sequences" tensor
        counts_min = tf.expand_dims(
            tf.expand_dims(
                tf.where(
                    tm.reduce_any(counts < -self.eps, axis=1),
                    -np.inf  * tf.ones(counts.shape[:1]),
                    tf.zeros(counts.shape[:1])
                    ),
            1), 2)
        counts_min = self.clip_invalid_counts * counts_min
        
        # Calculate scores for all predictions of all candidates
        # from the parent sequence score and the new prediction scores.
        # Flatten the shape (n*k, 1, tokens) array into a (n, k x tokens) array
        scores_y = (tf.expand_dims(tf.reshape(scores, y.shape[:-1]), 2) + 
                    tm.log(y) + 
                    self.pad_mask +
                    counts_min)
        
        
        
        
        
        y_ = tf.reshape(y, (self.n, self.k, 1, -1))
        y_sum = tf.cumsum(y_, axis=3)
        y_quantile = self.generator.uniform((self.n, self.k, 1, 1))
        y_sum = tf.where(y_sum > y_quantile, 1 - y_sum, 0)
        yscore_sum_useless, ysel = tf.math.top_k(y_sum)
        ysequence = tf.range(self.n * self.k)
        ymax = tf.reshape(ysel, (-1,))
            
        scores_this_step = tf.reshape(
            tf.gather(y_, ysel, batch_dims=3),
            (-1,)
            )
        scores = scores + tm.log(scores_this_step)
        # states is unchanged, since every sequence continues individually
        
        return ymax, ysequence, states, scores
    
    
    
    @tf.function
    def beam_traceback(self, sequences, y, scores, reverse = True):
        '''
        Traceback string sequences from a prediction result.
        
        Here, we actually just trace forward from zero.
        
        Note that since we force the generation of a large number
        of sequences by continuing to extend other sequences
        when the padding character is reached, the actual number
        of sequences per candidate is much larger than the specified k.
    
        Parameters
        ----------
        sequences : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        scores : TYPE
            DESCRIPTION.
    
        Returns
        -------
        traces : TYPE
            DESCRIPTION.
    
        '''
        
        
        ends = (tf.where(y == self.final_char,
                          tf.ones_like(y,  dtype='float32'),
                          tf.ones_like(y, dtype='float32') * np.Inf,
                          ) *
                (tf.reshape(tf.range(
                    tf.shape(y)[0],
                    dtype='float32'),
                    (-1,1)
                    ) + 1))
        
        positions = tf.argmin(ends, axis=0, output_type='int32')
        max_position = tf.maximum(tf.shape(y)[0], 
                                  tf.reduce_max(positions))
        score = tf.gather(tf.transpose(scores),
                          positions, 
                          batch_dims = 1)
        
        return tf.transpose(y), score, positions
        


    
    