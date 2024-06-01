# -*- coding: utf-8 -*-

from .sampling import *
from fp_management import fingerprinting as fpr
from fp_management import fingerprint_map as fpm
from fp_management import database as db
import pickle
from enum import IntEnum
import pathlib

class Bitmask(IntEnum):
    PREDICTED = 1
    TRUTH = 2
    TP = 4
    FP = 8
    TN = 16
    FN = 32

class SamplerFactory:
    def __init__(self, config):
        self.db_path = config['db_path_sampler']
        self.bitmatrix_path = pathlib.Path(self.db_path["path"]).with_suffix(".pkl")
        self.selected_fold = config['cv_fold']
        self.sampler_config = {}
        if 'sampler_config' in config.keys():
            self.sampler_config.update(config["sampler_config"])
        self._loaded = False
        self.db  = db.FpDatabase.load_from_config(self.db_path)
        self.ids = None
        self.bitmatrix_data = None
        self.bitmatrix_stats = None
    
    def load(self):
        
        ids_query_ = self.db.sql(
            "SELECT id from compounds WHERE"
            f" grp NOT IN ('invalid', 'fold{self.selected_fold}')"
            )
        # convert database 1-based indices to numpy 0-based indices
        self.ids = [x["id"] - 1 for x in ids_query_]
        # if(len(self.ids) < 20):
        #     print(str(self.ids))
        self.ids.sort()
        
        with open(self.bitmatrix_path, 'rb') as f:
            bitmatrix_data_ = pickle.load(f)
            self.bitmatrix_data = bitmatrix_data_[self.ids, :]
            self.bitmatrix_stats = self.fingerprint_bit_stats(self.bitmatrix_data)

    def get_sampler(self):
        if not self._loaded:
            self.load()
        sampler = BasicTpfpSampler(self.sampler_config)
        return sampler
    
    def round_fingerprint_inference(self):
        return True
    
    @staticmethod


    class Bitmask(IntEnum):
        PREDICTED = 1
        TRUTH = 2
        TP = 4
        FP = 8
        TN = 16
        FN = 32

    def fingerprint_bit_stats(self, bitmask_matrix):
        """
        Calculate the prediction stats for a bitmask matrix,
        (which has to be subsetted first because this is how we can do the 10CV.)
        """
        TP = np.sum(Bitmask.TP & bitmask_matrix, axis=0) // Bitmask.TP
        FP = np.sum(Bitmask.FP & bitmask_matrix, axis=0) // Bitmask.FP
        TN = np.sum(Bitmask.TN & bitmask_matrix, axis=0) // Bitmask.TN
        FN = np.sum(Bitmask.FN & bitmask_matrix, axis=0) // Bitmask.FN

        # The probability to sample sim-value 1 for a bit with truth-value 0
        # should be the false positive rate
        sampling_0 = FP / (TN + FP  )#+ 1)
        # The probability to sample sim-value 1 for a bit with truth-value 1
        # should be the recall
        sampling_1 = TP / (TP + FN ) # + 1)
        sampling = np.stack([sampling_0, sampling_1])

        return sampling
        

class BasicTpfpSampler(Sampler):
    def __init__(self, config = None,
                generator = tf.random.experimental.get_global_generator()):
        '''
        Parameters
        ----------
        bitmatrix_stats: a 2xn numpy int array, where n is the number of fingerprint bits.

            Every value is the probability of sampling a One (True) value,
            NOT the probability of sampling the correct value.

            row 0 is the probability of sampling a One for a Zero (ideally very low),
            row 1 is the probability of sampling a One for a One, (ideally very high)
            

        generator: tf.random.Generator
        Returns
        -------
        None.

        '''
        config_ = {
            'unchanged_rate': 0.01,
            'tpr': 0.7,
            'fpr': 0.1
        }
        if config is not None: 
            config_.update(config)

        self.tpr = config_["tpr"]
        self.fpr = config_["fpr"]
        Sampler.__init__(self)
        #self.stats = stats
        self.generator = generator
        # Proportion of fingerprints that should be passed through "unchanged" i.e. perfect
        self.unchanged_rate = config_["unchanged_rate"]

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

        unchanged = tf.expand_dims(
            tf.cast(
                self.generator.uniform(tf.shape(fp)[0:1], dtype="float32") < self.unchanged_rate,
                "float32"),
            1
            )

        #print(rand.numpy())
        # bits_x1 are the sampling results where x_j,i = 1
        bits_x0 = tf.cast(rand < self.fpr, "float32")
        bits_x1 = tf.cast(rand < self.tpr, "float32")

        bits_sampled = fp * bits_x1 + (1-fp) * bits_x0
        #bits_masked = bits_unmasked
        bits_with_unchanged = (1-unchanged) * bits_sampled + unchanged * fp

        return bits_with_unchanged
    
    @staticmethod
    def tanimoto(fp1, fp2):
        '''
        Calculates the tanimoto similarity between two equally-sized blocks
        of fingerprints, with samples in rows.
                
        '''
        # Some code from:
        # https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
        # Ref: salehi17, "Twersky loss function for image segmentation using 3D FCDN"
        # -> the score is computed for each class separately and then summed
        # alpha=beta=0.5 : dice coefficient
        # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
        # alpha+beta=1   : produces set of F*-scores
        # implemented by E. Moebel, 06/04/18
        
        alpha = 1
        beta = 1
        
        fp_1 = tf.cast(fp1, 'uint8')
        fp_2 = tf.cast(fp2, 'uint8')

        fp11 = fp_1 * fp_2
        fp10 = fp_1 * (1-fp_2)
        #fp00 = (1-fp_1) *(1-fp_2)
        fp01 = (1-fp_1) * fp_2
        
        c11 = tf.reduce_sum(fp11, axis=1)
        c10 = tf.reduce_sum(fp10, axis=1)
        c01 = tf.reduce_sum(fp01, axis=1)

        tanimoto = c11 / (c11 + alpha * c10 + beta * c01)
        return tf.cast(tanimoto, "float32")
        
        
        