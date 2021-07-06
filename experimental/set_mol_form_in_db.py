# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:01:57 2020

@author: stravsm
"""
from tqdm import tqdm

fp_db = db.FpDatabase.load_from_config(sc.config["db_path"])
allmols = fp_db.get_all()
mol = allmols[0]
for mol in tqdm(allmols):
    mol_id = mol["id"]
    mol_m = pickle.loads(mol["mol"])
    if mol_m is not '':
        if mol_m is not None:
            mf_text = rdMolDescriptors.CalcMolFormula(mol_m)
            fp_db.sql(f"UPDATE compounds SET mf_text = '{mf_text}' WHERE id = {mol_id}")


fp_db.close()
