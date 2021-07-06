# -*- coding: utf-8 -*-
"""
Created on 10.8.2020

@author: stravsm

Implements the Gumbel Top-k stochastic beam search by Kool et al.
arXiv:1903.06059v2

"""



import smiles_process as sp
from tensorflow import strings as ts
from tensorflow import math as tm
import tensorflow as tf
import numpy as np
import pandas as pd

from .decoder_base import DecoderBase
from .beam_search_decoder import BeamSearchDecoder



class GumbelBeamSearchDecoder(BeamSearchDecoder):
    
    # Gumbel computations adapted from 
    # https://github.com/wouterkool/stochastic-beam-search/blob/stochastic-beam-search/fairseq/gumbel.py
    @tf.function
    def gumbel_like(self, X):
        return self._gumbel(self.generator.uniform(shape=tf.shape(X)))
    
    @tf.function
    def gumbel(self, shape):
        return self._gumbel(self.generator.uniform(shape=shape))

    @staticmethod
    def _gumbel(u):
        return -tm.log(-tm.log(u))
    
    @tf.function    
    def gumbel_with_maximum(self, phi, T, axis=-1):
        """
        Samples a set of gumbels which are conditioned on having a maximum along a dimension
        phi.max(dim)[0] should be broadcastable with the desired maximum T
        """
        # Gumbel with location phi
        g_phi = phi + self.gumbel_like(phi)
        # TODO: this is inefficient, find argmax and extract the max instead
        # (or does TF optimize this internally?)
        Z = tf.reduce_max(g_phi, axis=axis)
        g = self._shift_gumbel_maximum(g_phi, T, axis, Z=Z)
        # CHECK_VALIDITY = True
        # if CHECK_VALIDITY:
        #     g_inv = _shift_gumbel_maximum(g, Z, axis)
        #     assert (((g_phi - g_inv) < 1e-3) | (g_phi == g_inv)).all()
        return g

    @staticmethod
    def shift_gumbel_maximum(g_phi, T, Z):
        u = T - g_phi + tm.log1p(-tf.exp(g_phi - Z))
        return T - tf.maximum(0., u) - tm.log1p(tf.exp(-tf.abs(u)))
    

    
    
    final_char = sp.smiles_ctoy(sp.FINAL_CHAR)
    initial_char = sp.smiles_ctoy(sp.INITIAL_CHAR)
    pad_char = sp.smiles_ctoy(sp.PAD_CHAR)
    final_char_c = sp.FINAL_CHAR
    
    def __init__(self, encoder, decoder, steps, n, k, kk, 
                 temperature = 1.,
                 config = {},
                 generator = tf.random.experimental.get_global_generator()):
        super(GumbelBeamSearchDecoder, self).__init__(encoder, decoder, steps, n, k, kk, temperature,
                                                      config = config)
        self.generator = generator


    @tf.function
    def beam_step(self, y, states, scores, scores_perturbed, counts):
        '''
        Performs the beam search priorization for one decoding step.
        In (for now for a single query sequence) the k (beam width) states and
        results from step n, as well as the cumulated, non-penalized scores for
        every of k "parents". Returns the k next query sequences and states
    
        Parameters
        ----------
        y : Tensor (n*k, 1, y_tokens)
            The prediction results for k beam candidates from step i prediction
        states : Tensor (n*k, states_size)
            The predicted states of k beam candidates after step i prediction
        scores : Tensor (n*k)
            Cumulated scores of candidates 0..k-1 at step i-1, going into step
            n prediction
        counts: Tensor (n*k, 1, (element_tokens + grammar_tokens))
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
            tf.where(
                tm.reduce_any(counts < -self.eps, axis=2),
                -np.inf  * tf.ones(counts.shape[:2]),
                tf.zeros(counts.shape[:2])
                ),
            2)
        # Calculate scores for all predictions of all candidates
        # from the parent sequence score and the new prediction scores.
        # Flatten the shape (n*k, 1, tokens) array into a (n, k x tokens) array
        scores_y = tf.reshape(
            (tf.expand_dims(tf.reshape(scores, y.shape[:-1]), 2) + 
                            tm.log(y) + 
                            self.pad_mask +
                            counts_min),
            (self.n, -1)
            )
                            
        
        # Add gumbel noise to all children
        scores_yg = scores_y + self.gumbel_like(scores_y)
        # Find maximum for each n
        yg_max = tf.reduce_max(scores_yg, axis=1, keepdims=True)
        
        scores_ygg = self.shift_gumbel_maximum(
            tf.reshape(scores_yg, (self.n, self.k, -1)),
            tf.reshape(scores_perturbed, (self.n, self.k, 1)),
            tf.reshape(yg_max, (self.n, 1, 1))
            )
        scores_ygg = tf.where(tf.math.is_nan(scores_ygg),
                              -np.inf, scores_ygg)
        
        scores_ygg_ = tf.reshape(scores_ygg, (self.n, -1))
        # Pick the top-k scores per sequence
        top_k = tm.top_k(scores_ygg_, self.k, sorted=False)
        # Recalculate the array indices for top_k. 
        # tm.top_k returns an array of shape (n, k) ranging up to 'k*tokens'.
        # * NOTE! The first "k" comes from the 'k' argument in tm.top_k.
        # * The second k comes from the k INPUT sequences per n-candidate
        #   in scores_y.
        # In the original y array, there are n*k sequences with 'tokens' prediction
        # candidates. In a flattened array, for the i-th sequence, the index starts
        # from i * (k * tokens), with (k*tokens) coincidentally already being the
        # tensor axis 1 length.
        top_k_index = tf.reshape(
            top_k[1] + tf.reshape(tf.range(self.n), (-1, 1)) * scores_y.shape[1], [-1])
        # And using the builtlin unravel_index, we transform this into
        # [parent_sequence, y_prediction] in the y array, which is 
        # (n*k, 1, tokens) shaped.
        ysequence = top_k_index // y.shape[2]
        ymax = top_k_index % y.shape[2]
        # Gather the precursor states and scores for the next prediction, which are 
        # the states and scores of the parent sequences after priorization
        # (and for the scores conveniently returned by top_k directly)
        states = tf.gather(states, ysequence)
        scores_perturbed = tf.reshape(top_k[0], [-1])
        scores = tf.gather_nd(scores_y, tf.stack(
            [tf.repeat(tf.reshape(tf.range(self.n), (-1, 1)), self.k),
            tf.reshape(top_k[1], (-1,))], axis=1)
            )
        #new_scores = tf.
        return ymax, ysequence, states, scores, scores_perturbed



    
    
    @tf.function
    def decode_beam(self, states_init):
        xstep = self.tokenization.embed_ytox(self.y_init)
        scores = self.scores_init
        scores_perturbed = self.scores_init
        
        states = states_init
        y_chain = tf.TensorArray(dtype="int32", size=self.steps)
        sequences_chain = tf.TensorArray(dtype="int32", size=self.steps)
        scores_chain = tf.TensorArray(dtype="float32", size=self.steps)
        scores_perturbed_chain = tf.TensorArray(dtype="float32", size=self.steps)
        i = 0
        # tf.while_loop explicitly
        beam_steps_cond = lambda i, y_, seq_, sc_, sp_, xstep, states, scores, scores_perturbed: i < self.steps
        def decode_beam_steps_body(i, y_, seq_, sc_, sp_, xstep, states, scores, scores_perturbed):
            y, states, counts = self.decoder([xstep, states], training=False)
            y = self.softmax(y)
            ymax, ysequence, states, scores, scores_perturbed = self.beam_step(y, states, scores, scores_perturbed, counts)
            xstep = self.tokenization.embed_ytox(ymax)
            y_ = y_.write(i, ymax)
            seq_ = seq_.write(i, ysequence)
            sc_= sc_.write(i, scores)
            sp_= sp_.write(i, scores_perturbed)
            i = i + 1
            return i, y_, seq_, sc_, sp_, xstep, states, scores, scores_perturbed
        
        _, y_chain, sequences_chain, scores_chain, scores_perturbed_chain, _, _, _, _ = tf.while_loop(
                cond = beam_steps_cond,
                body = decode_beam_steps_body,
                loop_vars = [i, y_chain, sequences_chain, scores_chain, scores_perturbed_chain,
                             xstep, states, scores, scores_perturbed],
                back_prop = True
                )
        sequences_final = sequences_chain.stack()
        y_final = y_chain.stack()
        scores_final = scores_chain.stack()
        scores_perturbed_final = scores_perturbed_chain.stack()
        # Return
        return sequences_final, y_final, scores_perturbed_final
    
    


    
    