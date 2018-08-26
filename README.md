# Inverse Fingerprinting RNN model
Michele Stravs, github.com/meowcat

## Input data
Using 500000 ZINC SMILES as a full dataset

## Processing

According to the specified yaml config file (by default using `config.yaml`
in the working directory)

1. smiles_prepare.py extracts a subset of `subset_size` SMILES from the 
    full dataset `smiles_full`
    and stores it on disk to `smiles_subset`
2. smiles_fingerprint.py loads the `smiles_subset`, computes RDK, Avalon
    and MACCS fingerprints and stores them in a pickle `smiles_fingerprints`
3. smiles_transform.py loads the `smiles_subset`, and generates an appropriate
    RNN learning dataset (slicing the sequence into the constituting 
    subsequences, and performing categorical encoding of each character.)
    It is stored in `smiles_encoded`.
4. smiles_model.py loads the fingerprints and encoded smiles, and 
    trains an RNN model with specified cell type `cell` 
    (LSTM, CuDNNLSTM, GRU, CuDNNGRU), for `epoch` epochs with `batch_size`
    batch size.
    
smiles_process.py contains utility functions for SMILES encoding.