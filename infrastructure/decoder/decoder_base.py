# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:26:27 2020

@author: stravsm
"""


import smiles_process as sp
from tensorflow import strings as ts
from tensorflow import math as tm
import tensorflow as tf
import numpy as np
import pandas as pd
import importlib

class DecoderBase:
    
    def __init__(self, encoder, decoder, steps, n, k, kk, temperature = 1., config = {}):
        self.config = {
            'tokenization': 'smiles_process',
            'clip_invalid_counts': True,
            'sequence_length_correction': False
            }
        if "decoder_config" in config:
            config_ = config["decoder_config"]
            self.config.update(config_)
            
        self.tokenization = importlib.import_module(self.config['tokenization'])
        self.clip_invalid_counts = self.config['clip_invalid_counts']
        
        self.final_char = self.tokenization.ctoy(self.tokenization.FINAL_CHAR)
        self.initial_char = self.tokenization.ctoy(self.tokenization.INITIAL_CHAR)
        self.pad_char = self.tokenization.ctoy(self.tokenization.PAD_CHAR)
            
        self.encoder = encoder
        self.steps = steps
        self.n = n
        self.decoder = decoder
        self.k = k
        self.kk = kk
        self.y_tokens = tf.convert_to_tensor(self.tokenization.y_tokens)
        # Make the constant templates that are required to start decoding
        self.y_init, self.scores_init, self.pad_mask = self._init_templates()
        # Make the embedding matrix and the final character resolution array
        self.eps = 0.001
        self.temperature = temperature
        self.clip_invalid_counts_factor = 1. * self.clip_invalid_counts
        self.sequence_length_correction = self.config['sequence_length_correction']


        
        
    def softmax(self, logits):
        return tf.nn.softmax(logits / self.temperature)

    def _init_templates(self):
        raise NotImplementedError("Not available in abstract base class")
    
    
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
        
        raise NotImplementedError("Not available in abstract base class")
        
    
    
    @tf.function
    def decode_beam(self, states_init):
        xstep = self.tokenization.embed_ytox(self.y_init)
        scores = self.scores_init
        
        states = states_init['states']
        y_chain = tf.TensorArray(dtype="int32", size=self.steps)
        sequences_chain = tf.TensorArray(dtype="int32", size=self.steps)
        scores_chain = tf.TensorArray(dtype="float32", size=self.steps)
        i = 0
        # tf.while_loop explicitly
        beam_steps_cond = lambda i, y_, seq_, sc_, xstep, states, scores: i < self.steps
        def decode_beam_steps_body(i, y_, seq_, sc_, xstep, states, scores):
            decoder_out = self.decoder({'tokens_X': xstep, 
                                        'states': states},
                                       training=False)
            y = decoder_out['tokens_X']
            states = decoder_out['states']
            counts = decoder_out['counts']
            
            y = self.softmax(y)
            ymax, ysequence, states, scores = self.beam_step(y, states, scores, counts)
            xstep = self.tokenization.embed_ytox(ymax)
            y_ = y_.write(i, ymax)
            seq_ = seq_.write(i, ysequence)
            sc_= sc_.write(i, scores)
            i = i + 1
            return i, y_, seq_, sc_, xstep, states, scores
        
        _, y_chain, sequences_chain, scores_chain, _, _, _ = tf.while_loop(
                cond = beam_steps_cond,
                body = decode_beam_steps_body,
                loop_vars = [i, y_chain, sequences_chain, scores_chain,
                             xstep, states, scores],
                back_prop = True
                )
        sequences_final = sequences_chain.stack()
        y_final = y_chain.stack()
        scores_final = scores_chain.stack()
        # Return
        return sequences_final, y_final, scores_final
    
    
    @tf.function
    def beam_traceback(self, sequences, y, scores, reverse = True):
        '''
        Traceback string sequences from a prediction result.
        
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
        
        # Reshape-reorder-reshape y and scores 
        # so top_k can find the best candidates
        # for each of n queries.
                
        r_t_r = lambda x: tf.reshape(
                    tf.transpose(
                    tf.reshape(x, [-1, self.n, self.k]),
                    [1,2,0]),
                [self.n, -1])       
        y_ = r_t_r(y)
        sc_ = r_t_r(scores)
        # Find best sequences:
        # Substitute the scores matrix with -Inf wherever we don't have
        # an end-of-sequence character in the y matrix.
        # Then find the best kk (which may be larger than k) sequences
        # among the candidates.
        ends = tf.where(y_ == self.final_char,
                        sc_,
                        tf.fill(sc_.shape, -np.inf))
        top_kk = tm.top_k(ends, self.kk)
        # The chosen best scores are automatically in top_kk[0]
        scores = top_kk[0]
                
        # Translate the positions in top_kk[1] (which are in an (steps, k) tensor)
        # to the "beam position" [0..k) and "timestep position" [0..steps)
        # So for example top_kk_vec[:,0] = [90, 36],
        # so the final position for the top sequence for sequence 0 of query 0
        # is y[36, 90].
        # The 51-th best sequence for query 0, top_kk_vec[:, 50], is [100, 33].
        # I.e. the sequence starts from y[33, 100] backwards.
        # The 10-th-best sequence for query 2 (for k=128), 
        # top_kk_vec[:, 2*128+9] = top_kk_vec[:, 265] = [67,26],
        # i.e. the sequence tarts from y[26, 67 + 2*128] = y[26, 323] backwards.
        # In y[i, j], i is the timestep and j is the sequence.
        top_kk_vec = tf.unravel_index(
            tf.reshape(top_kk[1], [-1]),
            [self.k, self.steps])
        
        # Translate the "beam position" to the batch position for an n*k batch:
        # top_kk_vec[0, i] for i in [0, k*n) contains the sequence position
        # to start from. 
        # In the above example, 
        # pos[2 * 128+9] = pos[265] will now point to 67 + 2*128 = 323
        i_source = tf.range(self.n*self.kk) // self.kk
        pos = top_kk_vec[0] + self.k * i_source
        
        # step is the "timestep position" to start the "traceback to 0" from,
        # store it to crop the sequences later.
        step = top_kk_vec[1]
        length = step
        
        # tf.while loop:
        # Starting from steps "step", run i steps "backwards" in parallel
        # until all steps reach 0. 
        # At every step i, fill the TensorArray traces[i] with the ith-last
        # characters for every sequence (if a sequence reaches step 0,
        # continue to fill with the character 0),
        # and get the position of the preceding character for the sequence.
        max_length = tm.reduce_max(step)
        traces = tf.TensorArray(dtype='int32', size=max_length)
        def beam_traceback_body(i, pos, step, traces):
            token_at_pos = tf.gather_nd(y, tf.stack([step, pos], axis=1))
            continue_at = tf.gather_nd(sequences, tf.stack([step, pos], axis=1))
            # To simplify matters for understanding, fill with zero
            # if the sequence has reached the beginning
            
            
            traces = traces.write(i, token_at_pos)
            i = i + 1
            step = tm.maximum(
                step - 1,
                tf.zeros_like(step))
            return i, continue_at, step, traces
        def beam_traceback_cond(i, pos, step, traces):
            return tf.reduce_any(step > 0)
        
        i = 0
        _, _, _, traces = tf.while_loop(
            cond = beam_traceback_cond,
            body = beam_traceback_body,
            loop_vars = [i, pos, step, traces]
            )
        
        
        traces = tf.transpose(traces.stack())
        traces = tf.sequence_mask(length+1, max_length, dtype=tf.int32) * traces
        
        # The true length of the sequence is startingstep + 1,
        # but as this could exceed the last step, we have to cp it
        length_with_termination_capped = tf.minimum(max_length, length+1)
        
        # We now have the unreversed sequences, starting from the final
        # character running to the first character
        if not reverse:
            return traces, scores, length_with_termination_capped
                
        
        # How precisely does tf.reverse_sequence behave?
        # zero_to_nine = np.stack([np.arange(10) for i in range(7)])
        # tf.reverse_sequence(zero_to_nine, [4,4,3,3,2,2,1], 1)
        # <tf.Tensor: shape=(7, 10), dtype=int32, numpy=
        # array([[3, 2, 1, 0, 4, 5, 6, 7, 8, 9],
        #        [3, 2, 1, 0, 4, 5, 6, 7, 8, 9],
        #        [2, 1, 0, 3, 4, 5, 6, 7, 8, 9],
        #        [2, 1, 0, 3, 4, 5, 6, 7, 8, 9],
        #        [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        #        [1, 0, 2, 3, 4, 5, 6, 7, 8, 9],
        #        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])>
        traces = tf.reverse_sequence(
            traces,
            length_with_termination_capped,
            1
            )
        return traces, scores, length_with_termination_capped

        
        # Note: disregard the array after the terminating character!
        # Those are repeated values of y[0,i] from the beam_backtrace operation,
        # which continues to fill up the array with the first element until
        # all sequences are done.
        # traces_rev = tf.reverse_sequence(
        #     traces,
        #     step0,
        #     seq_axis=0,
        #     batch_axis=1
        #     )
        # # Transpose the resulting traces into (steps, n, k) shape
        # traces_rev = tf.transpose(
        #     tf.reshape(
        #         traces_rev,
        #         [-1, self.n, self.kk]),
        #     [1,2,0])
        
        # # return traces and associated scores, length
        # return traces_rev, top_kk[0], step0
    
    def sequence_ytoc(self, seq):
        return self.tokenization.sequence_ytoc(seq)

    
    def format_results(self, smiles, scores, **kwargs):
        seq_df = pd.DataFrame({
            "smiles": smiles,
            "score": np.reshape(scores, -1),
            "id": range(len(smiles))
            })
        seq_df["n"] = seq_df["id"] // self.kk
        seq_df["k"] = seq_df["id"] % self.kk
        for k, v in kwargs.items():
            seq_df[k] = v
        return seq_df
    
    def format_reference(self, smiles, fingerprint):
        seq_df = pd.DataFrame({
            "smiles": smiles,
            "score": np.inf,
            "id": range(len(smiles)),
            "n": range(len(smiles)),
            "k": -1,
            "fingerprint": fingerprint
            })
        return seq_df



    def score_step(self, y_pred, y_sequence, counts):
        '''
        Performs a step in sequence scoring under the model.
    
        Parameters
        ----------
        y_pred : Tensor k x 1 x y_tokens
            The prediction results for k sequence from step n prediction
        y_sequence: Tensor k x 1 x y_tokerns
            The actual sequence characters for k sequences at step n
        states : Tensor k x tensor_size
            The predicted states of k sequences after step n prediction
        scores : Tensor k x 1
            Cumulated scores of sequences 0..k-1 at step n-1, going into step
            n prediction
        counts: Tensor k x 1 x (element_tokens + grammar_tokens)
            Remaining element count. Kill the sequence if any goes <0.
        
        Returns
        -------
        scores : Tensor k x 1
            The cumulated scores after step n priorization
            (the input scores for step n+1 priorization)    
        ymax :
            for convenience, the chosen character token to feed forward
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
        
        # # Calculate scores for all predictions of all candidates
        # # from the parent sequence score and the new prediction scores.
        # # Flatten the shape (n*k, 1, tokens) array into a (n, k x tokens) array
        # scores_y = (tf.expand_dims(tf.reshape(scores, y.shape[:-1]), 2) + 
        #             tm.log(y) + 
        #             self.pad_mask +
        #             counts_min)
        # scores_y = tf.reshape(scores_y, [self.n, -1])
        
        
        # # Make a y-shaped "kill invalid sequences" tensor
        # counts_min = tf.expand_dims(tf.where(
        #         tm.reduce_any(counts < -self.eps, axis=2),
        #         -np.inf  * tf.ones(tf.shape(counts)[:2]),
        #         tf.zeros(tf.shape(counts)[:2])
        #         ), 2)
        
        #y_pred = y_pred + self.pad_mask + counts_min
        # Get the score for the actually chosen sequence position
        # Note, there is only one non-zero position per sequence,
        # so reduce_max is OK; note that reduce_sum fails because
        # -inf * 0 = nan for the padding character
        scores_y = tf.squeeze(tf.reduce_max(
            y_pred * y_sequence,
            axis = 2
            ))
        #
        ymax = tf.reshape(tm.top_k(y_sequence)[1], (tf.shape(y_sequence)[0],))
        scores = tm.log(scores_y)
        
        return ymax, scores


    @tf.function(experimental_relax_shapes=True)
    def score_sequences(self, states_init, y_sequences):
        '''
        

        Parameters
        ----------
        states_init : Tensor (n, states_size)
            n initial states of state_size each
        sequences : Tensor (n, step_size, y_token_size)
            n one-hot encoded sequences with step_size number of steps

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        y_init_ = tf.ones(tf.shape(states_init['states'])[0], 'int32') * self.initial_char
        xstep = self.tokenization.embed_ytox(y_init_)
        scores = tf.zeros((tf.shape(states_init['states'])[0]))
        steps = tf.shape(y_sequences)[1]
        states = states_init['states']
        scores_chain = tf.TensorArray(dtype="float32", size=steps)
        i = 0
        # tf.while_loop explicitly
        score_steps_cond = lambda i, sc_, xstep, states: i < steps
        def score_steps_body(i, sc_, xstep, states):
            decoder_out = self.decoder({'tokens_X': xstep, 
                                        'states': states},
                                       training=False)
            
            y = decoder_out['tokens_X']
            states = decoder_out['states']
            counts = decoder_out['counts']
            
            y = self.softmax(y)
            y_seq_step = tf.expand_dims(y_sequences[:,i,:], 1)
            ymax, scores = self.score_step(y, y_seq_step, counts)
            xstep = self.tokenization.embed_ytox(ymax)
            sc_= sc_.write(i, scores)
            i = i + 1
            return i, sc_, xstep, states
        
        _, scores_chain, _, _ = tf.while_loop(
                cond = score_steps_cond,
                body = score_steps_body,
                loop_vars = [i, scores_chain,
                             xstep, states],
                back_prop = True
                )
        scores_final = tf.transpose(scores_chain.stack())
        # Return
        return scores_final


    
    