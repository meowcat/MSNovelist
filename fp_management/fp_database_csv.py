# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:12:44 2021

@author: stravsm
"""

from .fp_database import *

def fp_database_pickle(db_file, config):
    return pickle.load(open(db_file, 'rb'))
    

class FpDatabaseCsv(FpDatabase):
    
    def __init__(self, db_file, config):
        # Update the default config with the given config:
        config_ = {
            'fp_map': None,
            'nrows': None,
            'construct_from': "inchi",
            "random_seed": 44,
            'reload_smiles_pubchem': False,
            'scramble_fingerprints': False
            }
        config_.update(config)
        super().__init__(db_file, config_)

        self._init_dispatch(config_['fp_map'],
                            config_['nrows'],
                            config_['construct_from'],
                            config_["random_seed"],
                            config_['reload_smiles_pubchem'],
                            config_['scramble_fingerprints'])
        
    def dump_pickle(self, path):
        self.fingerprinter = None
        pickle.dump(self, open(path, 'wb'))

    def _init_dispatch(self, fp_map, nrows = None, 
                       construct_from = "inchi", 
                       random_seed = 44,
                       reload_smiles_pubchem = False,
                       scramble_fingerprints = False):
        '''
        Parameters
        ----------
        db_file : TYPE
            DESCRIPTION.
        fp_map : int
            Note: this needs to be the "long" fingerprint map - 3xxx bits from
            CANOPUS / CSI:FID 4.4) which is in column 1 of the "statistics" csv
        nrows: int
            If given, limits how many records are read - for e.g. unittest
            purposes
        Returns
        -------
        None.

        '''
        self.data_grp = {}
        # read information block
        self.data_information = pd.read_table(
            self.db_file, delimiter='\t', 
            header=None,
            usecols = (0,1,2,3), 
            names=("id","inchikey", construct_from ,"fingerprint"), 
            nrows = nrows,
            comment = None)
        self.data_information.set_index("id", inplace=True)
        self.fp_map = fpm.FingerprintMap(fp_map)
        self.fp_len = len(self.data_information["fingerprint"][0])
        self.fp_real_len = int(max(self.fp_map.positions)+1)
        self.random_seed = random_seed
        
        # Read predicted fingerprints
        data_fp_predicted = np.genfromtxt(
            self.db_file, 
            delimiter='\t', 
            names=None, 
            comments = None,
            usecols= tuple(map(lambda x: x + 4,
                              range(self.fp_len)
                              )),
            max_rows = nrows
            )
        
        if reload_smiles_pubchem:
            self.data_information = get_smiles_pubchem(self.data_information, db_pubchem)
            self.data_information["smiles"] = self.data_information["smiles_in"]
            
        
        self.fingerprinter = fpr.Fingerprinter.get_instance()
        
        self.construct_from = construct_from
        # Read real fingerprints
        data_fp_true = np.array([list(map(int, x)) 
                                 for x in self.data_information["fingerprint"]])

        # Reshape predicted 3541 FP into the full 7593 bit FP
        data_fp_full = np.zeros((data_fp_predicted.shape[0],
                                 self.fp_real_len))
        data_fp_full[:,self.fp_map.positions] = data_fp_predicted
        # data_fp_realigned = np.array([
        #     fpr.realign_fp_numpy(i) for i in list(data_fp_full)])
        self.data_fp_predicted = data_fp_full
        # Reshape true 3541 FP into the full 7593 bit FP
        data_fp_full = np.zeros((data_fp_true.shape[0],
                                 self.fp_real_len))
        data_fp_full[:,self.fp_map.positions] = data_fp_true
        # data_fp_realigned = np.array([
        #     fpr.realign_fp_numpy(i) for i in list(data_fp_full)])
        self.data_fp_true = data_fp_full
        
        
        self.data_information["perm_order"] = 0
        self.data_information["source"] = ''
        self.data_information["grp"] = ''
        self.data_information["row_id"] = np.arange(
            len(self.data_information), dtype="int32")
        
        # For now, we can only scramble predicted fingerprints,
        # because we need the true ones for comparison, right?
        # Or are those pulled from the fingerprinter?
        
        self.scramble_fingerprints = scramble_fingerprints
        if self.scramble_fingerprints:
            fp_order = np.arange(data_fp_predicted.shape[0])
            np.random.shuffle(fp_order)
            data_fp_predicted = data_fp_predicted[fp_order,:]
            self.data_fp_predicted = data_fp_predicted
            
        self.process_smiles()
        self.randomize(self.random_seed)

    def process_smiles(self):
        
        self.data_information = process_df(
            self.data_information,
            self.fingerprinter,
            construct_from = self.construct_from,
            write = ["smiles_generic", "smiles_canonical", "inchikey", "inchikey1","mf", "mol"],
            block_id = None)
            
    def close(self):
        pass
    
    def set_grp(self, name, table, fold = False):
        '''
        

        Parameters
        ----------
        name : Name of the group to be created
            DESCRIPTION.
        table : Pandas dataframe with an "id" column - or the first column
            will be used - to select the database items to be assigned to
            this group.
        fold: optional
            If True, then not one group but n groups will be created, with
            the name as a prefix and the fold id - derived from a "fold" 
            column in the 'table' - as a suffix.
        Returns
        -------
        None.

        '''
        if not any(table.columns == "id"):
            table["id"] = table.iloc[:,0]
        if not fold:
            self.data_grp.update({
                    name: table["id"]
                })
            # Verify that the group is completely present,
            # otherwise this will fail:
            #_ = self.get_grp(name)
        else:
            # Make folds
            folds = set(table["fold"])
            for fold in folds:
                self.data_grp.update({
                    name + str(fold): table.loc[table["fold"] == fold]
                    })
                #_ = self.get_grp(name + str(fold))
    
    
    def get_grp(self, grp):
        '''
        untested

        Parameters
        ----------
        grp : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if grp in self.data_grp:
            grp_ok = self.data_information.reindex(self.data_grp[grp])
            missing = grp_ok["row_id"].isna()
            grp_ok = grp_ok.copy()
            grp_ok["grp"] = grp
            if any(missing):
                n_missing = sum(missing)
                warn(f"Not all entries from this group are present: {n_missing} missing")
        else:
            grp_match = map(lambda s: s.startswith(grp), self.data_information.index)
            grp_ok = self.data_information.loc[grp_match].copy()
            grp_ok["grp"] = grp
            

        
        #return grp_ok
        return [
            self._record_iter(x)
            for x in grp_ok.loc[grp_ok["row_id"].notna()].sort_values("perm_order").itertuples()
            ]
            
    def get_all(self):
        return [self._record_iter(x) 
                for x in 
                self.data_information.sort_values("perm_order").itertuples()]
        
    def _record_iter(self, record):
        d = record._asdict()
        row_id = int(d["row_id"]) # this is very ugly, because
        # row_id gets transformed to float during reindex() in get_grp()
        # when NA are introduced
        #print(row_id)
        d.update({
            'fingerprint': self.data_fp_true[row_id,:],
            "fingerprint_degraded": self.data_fp_predicted[row_id,:]
            })
        return d
                   
    def randomize(self, keep = True, random_seed = 45):
        if random_seed is not None:
            seed(random_seed)
        self.data_information["perm_order"] = \
            [random() for _ in range(self.data_information.shape[0])]
            
    def get_pipeline_options(self):
        
        options = super().get_pipeline_options()
        options['unpack'] = False
        options['unpickle_mf'] = False
        return options
    
FpDatabase.register_mapping(".csv", FpDatabaseCsv)
FpDatabase.register_mapping(".pkl", fp_database_pickle)