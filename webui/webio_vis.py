import pywebio
from pywebio import *
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio import session

from pyteomics import mgf
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus

import processing
import tempfile
import matplotlib.pyplot as plt
import pathlib

import sys
import os
import re
sys.path.append(os.environ['MSNOVELIST_BASE'])

import subprocess
import time

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
import pandas as pd
import pickle
from tqdm import tqdm

import smiles_config as sc

mol_width = 300
mol_height = 200

# https://iwatobipen.wordpress.com/2019/02/16/make-interactive-dashboard-with-dash2-chemoinformatcs-rdkit/
def smi2svg(smi, width, height):
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(width,height)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    return svg.encode('utf-8')

def visualize(results_file):

    results = pickle.load(open(results_file, 'rb'))

    mol_stack = {}

    results["mf_text"] = [rdMolDescriptors.CalcMolFormula(m) for m in tqdm(results["mol"])]
    results["mz"] = [rdMolDescriptors.CalcExactMolWt(m) + 1.0072 for m in tqdm(results["mol"])]
    results["legend"] = list(map(
        lambda x: "{inchikey1}, m/z {mz:.4f}".format(**x[1]), 
        results.iterrows()))

    def mol_popup(name):
        popup(
            name,
            [
                put_image(Draw.MolToImage(mol_stack[name], size = (mol_width, mol_height))),
            ]
            )

    def table_topresults(score, mf):

        results_show = results.copy()
        if mf == 'best_mf':
            results_show.sort_values(['n', score], ascending = [True, False], inplace = True)
            results_g = results_show.groupby('query')
        if mf == 'best_score':
            results_show.sort_values([score], ascending = [False], inplace = True)
            results_g = results_show.groupby('query')
        if mf == 'all_mf':
            results_show.sort_values([score], ascending = [False], inplace = True)
            results_g = results_show.groupby(['query', 'n'])
        results_top = results_g.head(1).copy()
        results_top.sort_values('m', inplace = True)
        results_blocks = results_top.groupby('query')

        table_out = []
        for query, table_block in results_blocks:
            table_out.append(
                [span(put_markdown(f"**{query}**"), col = 5)]
            )
            for i, table_row in table_block.iterrows():

                mol_stack[table_row["inchikey1"]] = table_row["mol"]

                output_row = [
                    table_row['mf_text'],
                    table_row['legend'],
                    f'{table_row[score]:.2f}',
                    put_column([
                        put_image(
                            Draw.MolToImage(table_row["mol"], size = (mol_width, mol_height))
                        ),
                        put_buttons(
                            [{'label': 'view', 'value': table_row["inchikey1"]}], 
                            onclick = lambda inchikey : mol_popup(inchikey), 
                            small = True, link_style = True)
                    ]),
                    table_row['smiles'],
                ]
                table_out.append(output_row)
        return table_out

    settings = input_group(
        'Result display settings',
        [
            radio(
                label = 'Formula',
                options = [
                    {'label': 'Show best-fitting formula (default)', 'value': 'best_mf', 'selected': True},
                    {'label': 'Show formula with highest-scoring structure', 'value': 'best_score'},
                    {'label': 'Show highest-scoring structure for each formula', 'value': 'all_mf'},
                ],
                name = 'mf_selection'
            ),
            select(
                label = 'Sort by', 
                options = [
                    {'label': 'ModPlatt score', 'value': 'score_mod_platt'},
                    {'label': 'RNN score', 'value': 'score_decoder'},
                ],
                name = 'score')
        ]
    )

    table_results = table_topresults(
        settings['score'],
        settings['mf_selection']
    )

    put_table(table_results,
        header = ['formula', 'info', 'score', 'molecule', 'smiles'])

    

