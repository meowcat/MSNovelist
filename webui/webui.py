# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:51:42 2020

@author: stravsm
"""


import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import smiles_config as sc

from tqdm import tqdm
from IPython.display import SVG

import dash
import dash_table
import pandas as pd
from dash_table.Format import Format, Scheme, Sign, Symbol
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64

from rdkit.Chem import Draw
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor

import processing

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
    return svg


# Load results
results_path = sc.config['webui_results']

results = pickle.load(open(results_path, 'rb'))

# Reformat and split results
results["mf_text"] = [rdMolDescriptors.CalcMolFormula(m) for m in tqdm(results["mol"])]
results["mz"] = [rdMolDescriptors.CalcExactMolWt(m) + 1.0072 for m in tqdm(results["mol"])]
results["legend"] = list(map(
    lambda x: "{inchikey1}, {mz:.4f}, {mf_text} ({score:.2f})".format(**x[1]), 
    results.iterrows()))
results.sort_values('score', ascending=False, inplace=True)
results_by_query = {m: table for m, table in results.groupby('query')}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

results_columns ={i: {"name": i, "id": i} for i in 
                  ["id", "mz", "mf_text", "score", "score_lim_mod_platt", "inchikey1", "smiles"]}
results_columns["mz"].update({
    'type': 'numeric',
    'format': Format(precision = 4, scheme = Scheme.fixed)
    })
results_columns["score"].update({
    'type': 'numeric',
    'format': Format(precision = 2, scheme = Scheme.fixed)
    })
results_columns["score_lim_mod_platt"].update({
    'type': 'numeric',
    'format': Format(precision = 2, scheme = Scheme.fixed)
    })

# Define UI
app.layout = dbc.Container([
    html.H1("MSNovelist prediction results"),
    dbc.Tabs([
    dbc.Tab(label = "input", children = [
        "Upload MGF files for analysis",
         dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
        
        ]),
    dbc.Tab(label = "results", children = [

        # Row 2: half-width query and MF selector
        dbc.Row([
            dbc.Col([
                "Select molecule query: ",
                dcc.Dropdown(
                    id = "select_molecule",
                    options = [{'label': x, 'value': x} for x in results_by_query.keys()],
                    value = None
                ),
                "Select molecular formula: ",
                dcc.Dropdown(
                    id = "select_formula",
                    options = [{'label': "*", 'value': 'all'}],
                    value = 'all'
                    )
                ], width = 6),
            dbc.Col([
                html.Img(
                    id='molecule_render',
                    width = 400,
                    height = 250
                    )], width=6)
            ]),
        # Row 3: results table
        dbc.Row([
            dbc.Col([
                dash_table.DataTable(
                    id='results_table', 
                    columns = list(results_columns.values()),
                    sort_action = "native",
                    row_selectable = 'single')
                ])
            ])
    ])
    ])])


# Define callbacks

@app.callback(
    Output(component_id = "select_formula", component_property = "options"),
    [Input(component_id = "select_molecule", component_property = "value")]
    )
def update_select_formula(query):
    if query is None:
        return []
    selected_results = results_by_query[query]
    formulas = selected_results.groupby("mf_text").size()
    size_tot = len(selected_results)
    formulas_options = [{'value': "*", 'label': f"all ({size_tot})"}]
    formulas_options.extend([
        {'value': mf, 'label': f"{mf} ({size})"} for mf, size in formulas.items()])
    return formulas_options

@app.callback(
    Output(component_id = "results_table", component_property = "data"),
    [Input(component_id = "select_molecule", component_property = "value"),
     Input(component_id = "select_formula", component_property = "value")]
    )
def update_results_table(query, mf):
    if query is None:
        return [{}]
    if mf is None:
        return [{}]
    return _update_results_table(query, mf).to_dict("records")

def _update_results_table(query, mf):

    selected_results = results_by_query[query]
    selected_results = selected_results[list(results_columns.keys())]
    if mf != "*":
        selected_results = selected_results.loc[selected_results.mf_text == mf]
    return selected_results


@app.callback(
    Output(component_id = "molecule_render", component_property = "src"),
    [Input(component_id = "select_molecule", component_property = "value"),
     Input(component_id = "select_formula", component_property = "value"),
    Input("results_table", "selected_rows")])
def update_render_molecule(query, mf, mol):
    if query is None:
        return ""
    if mf is None:
        return ""
    if mol is None:
        return ""
    table = _update_results_table(query, mf)
    smiles = table["smiles"].iloc[mol[0]]
    svg_ = smi2svg(smiles, 400, 250)
    encoded = base64.b64encode(svg_.encode()) 
    svg = 'data:image/svg+xml;base64,{}'.format(encoded.decode())
    return svg

# update_select_formula(0)
# update_results_table(0, "*")



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)