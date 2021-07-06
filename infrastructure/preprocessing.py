# -*- coding: utf-8 -*-


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors

from enum import Enum, auto

from chempy import Substance
import chempy


ELEMENTS = ['C','F','H','I','Cl','N','O','P','Br','S']
ELEMENTS_chempy = [chempy.util.periodic.atomic_number(e) for e in ELEMENTS]

class SubstanceCheck:#(Enum):
    OK = 0#auto()
    SMILES_DOT = 1#auto()
    SMILES_LEN = 2#auto()
    SMILES_OK = 3#auto()
    MOL_WT = 4#auto()
    MOL_CHARGE = 5#auto()
    MOL_ELEMENTS = 6#auto()
    EXCEPTION = 7

def substance_OK(s):
    try:
        if len(s) > 127:
            return SubstanceCheck.SMILES_LEN
        if s.find('.') != -1:
            return SubstanceCheck.SMILES_DOT
        m = Chem.MolFromSmiles(s)
        if m is None:
            return SubstanceCheck.SMILES_OK
        mw = Descriptors.ExactMolWt(m)
        if mw > 1000:
            return SubstanceCheck.MOL_WT
        fc = Chem.rdmolops.GetFormalCharge(m)
        if fc != 0:
            return SubstanceCheck.MOL_CHARGE
        fo = rdMolDescriptors.CalcMolFormula(m)
        s_form = chempy.Substance.from_formula(fo)
        if not all([(e in ELEMENTS_chempy) for e in s_form.composition.keys()]):
            return SubstanceCheck.MOL_ELEMENTS
        return SubstanceCheck.OK
    except:
        return SubstanceCheck.EXCEPTION