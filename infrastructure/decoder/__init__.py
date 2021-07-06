# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:31:27 2020

@author: stravsm
"""

from .beam_search_decoder import BeamSearchDecoder
from .decoder_base import DecoderBase
from .gumbel_decoder import GumbelBeamSearchDecoder
from .stochastic_sampler import StochasticSampler


decoder_dict = {
    'beam_search': BeamSearchDecoder,
    'gumbel_beam_search': GumbelBeamSearchDecoder,
    'stochastic_sampler': StochasticSampler
    }

def get_decoder(name, **kwargs):
    return decoder_dict[name]
    
