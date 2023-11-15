
import pandas as pd
import numpy as np

class FingerprintMap:
    '''
    Container to load a fingerprint map descriptor from SIRIUS
    containing fingerprint absolute positions, and prediction statistics
    for every bit that are used in scoring and fingerprint sampling.
    '''

    def __init__(self, filename = None, data = None, subset = None, 
                 explicit_len = None):
        
        if filename is not None:
            data = pd.read_csv(filename, delimiter='\t')
        if data is None:
            raise ValueError("No file or dataframe supplied.")
        
        data.rename(columns = {'absoluteIndex': 'position'}, inplace = True)
        data.set_index("position", inplace = True)
        
        if subset is not None:
            self.subset_positions = subset
        else:
            self.subset_positions = data.index

        self.fp_len = explicit_len or np.max(self.subset_positions) + 1
        
        data["sens"] = (data.TP + 0.5) / (data.TP + data.FN + 1)
        data["spec"] = (data.TN + 0.5) / (data.FP + data.TN + 1)
        self.data = data
        self.build_tables()
    
    def build_tables(self):
        
        data_subset = self.data.loc[self.subset_positions]
        # extract fingerprint map and numpy stats for the scoring functions
        self.positions = data_subset.index.tolist()
        self.stats = data_subset[["sens", "spec", "F1"]].to_numpy(dtype='float32')
    
    def subset_map(self, subset, iloc = False):
        '''
        Select a new subset and build corresponding fp_map and stats tables

        Parameters
        ----------
        subset : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        if iloc:
            self.subset_positions = self.data.iloc[subset].index
        else:
            self.subset_positions = subset
        self.build_tables()
