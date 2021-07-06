# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:31:28 2020

@author: stravsm
"""


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
import tensorflow as tf
from tensorflow import math as tm
from tensorflow.keras import backend as K
import pickle

import os

class SaveOptimizerCallback(Callback):
    
    def __init__(self, weights_path):
        self.weights_path = weights_path
    
    def on_epoch_end(self, epoch, logs=None):

        symbolic_weights = getattr(self.model.optimizer, 'weights')
        epoch_ = epoch + 1
        optimizer_weights_path = os.path.join(
            self.weights_path,
            f"o-{epoch_:02d}.hdf5")
        weight_values = K.batch_get_value(symbolic_weights)
        with open(optimizer_weights_path, 'wb') as f:
            pickle.dump(weight_values, f)




# adapted from https://stackoverflow.com/a/47738812/1259675
class AdditionalValidationSet(Callback):
    def __init__(self, validation_dataset, validation_set_name, verbose=0):
        """
        

        Parameters
        ----------
        validation_dataset : tf.data.Dataset
            DESCRIPTION.
        validation_set_name : TYPE
            DESCRIPTION.
        verbose : TYPE, optional
            DESCRIPTION. The default is 0.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """

        super(AdditionalValidationSet, self).__init__()
        self.validation_dataset = validation_dataset
        self.validation_set_name = validation_set_name
        self.epoch = []
        self.history = {}
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        results = self.model.evaluate(self.validation_dataset,
                                      verbose=self.verbose,)
        
        for metrics_name, result in zip(self.model.metrics_names, results):
            valuename = self.validation_set_name + "_" + metrics_name
            self.history.setdefault(valuename, []).append(result)
            logs.setdefault(valuename, result)
            #self.model.history.history.setdefault(valuename, []).append(result)