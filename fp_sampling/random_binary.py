# -*- coding: utf-8 -*-

from .sampling import *
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm


class SamplerFactory:
    def __init__(self, config):
        self.base_folder = config["base_folder"]
        self.fp_map_ = config["fp_map"]
        self.f1_cutoff = config["f1_cutoff"]
    
    def get_sampler(self):
        fp_map = fpm.FingerprintMap(self.fp_map_)
        sampler = RandomBinarySampler(fp_map, self.f1_cutoff)
        return sampler
    
    def round_fingerprint_inference(self):
        return True

        

        
        


class RandomBinarySampler(Sampler):
    def __init__(self, fp_map, f1_cutoff = 0,
                 generator = tf.random.experimental.get_global_generator()):
        '''
        Parameters
        ----------
        stats : pandas.DataFrame
            DataFrame with columns ['position', 'TP', 'FP', 'FN, 'TN']
        fp_map : array-like
            A fingerprint map - an index of fingerprint positions of the 
            predicted fingerprint in the full (7xxx bit) fingerprint.
        generator: tf.random.Generator
        Returns
        -------
        None.

        '''
        Sampler.__init__(self)
        #self.stats = stats
        self.generator = generator
        self.fp_map = fp_map
        self.stats = self.fp_map.stats
        self.f1_cutoff = f1_cutoff
        self.f1_mask = self.stats[:,2] >= self.f1_cutoff
        self.f1_mask = self.f1_mask.astype("float32")
    
    @tf.function
    def sample(self, fp):
        '''
        Samples simulated predicted fingerprints Y from an array of true 
        fingerprints X [x_j, i] of shape (n, fp_len)
        (i.e. i is the bit in the fingerprint, j is the fingerprint in the batch)
        
        using binary random sampling according to the loaded prediction statistics.

        Parameters
        ----------
        fp : np.array of tf.Tensor(shape=(n, fp_len)) dtype=float32 but 
            Array of n fingerprints expected to be € {0, 1} (but of float type)

        Returns
        -------
        Equally-shaped tensor with simulated predicted binary fingerprint, i.e. 
            all values are € {0, 1]

        '''
        rand = self.generator.uniform(tf.shape(fp), dtype="float32")
        #print(rand.numpy())
        # bits_x1 are the sampling results where x_j,i = 1
        bits_x1 = tf.cast((rand < tf.expand_dims(self.stats[:,0], 0)), "float32")
        bits_x0 = tf.cast((rand > tf.expand_dims(self.stats[:,1], 0)), "float32")
        bits_unmasked = fp * bits_x1 + (1-fp) * bits_x0
        bits_masked = self.f1_mask * bits_unmasked
        return bits_masked
        
        
        