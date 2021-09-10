import pywebio
from pywebio import *
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *

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

import smiles_config as sc


def main():
    put_markdown("# MSNovelist web interface")

    set_scope('input_scope', position = 0)
    set_scope('output_scope', position = -1)

    # Upload and store spectra
    with use_scope('input_scope', clear=True):
        spectra_upload = input_group("Spectra upload", inputs = [
            file_upload(
                label = "Select MGF file",
                accept = ['.mgf'],
                name = "spectra"
            )
        ])

    target_path = os.path.join(
        sc.config['eval_folder'],
        'spectra.mgf'
    )

    with open(target_path, "wb") as f:
        f.write(spectra_upload["spectra"]["content"])


    # Visualize spectra
    spectra_ = mgf.read(target_path)
    spectra = [s for s in spectra_]
    spectra_vis = [processing.get_su_spectrum(s) for s in spectra]
    spectra_names = [s['params']['title'] for s in spectra]
    spectra_process = [True for s in spectra]

    # Review spectra settings

    def plot_spectrum(i):
        tf, spectrum_temp_png = tempfile.mkstemp(suffix = ".png")
        os.close(tf)
        spectrum_fig = plt.figure()
        spectrum_fig.add_subplot(sup.spectrum(spectra_vis[i])) 
        spectrum_fig.savefig(spectrum_temp_png)
        spectrum_img = open(spectrum_temp_png, 'rb').read()
        return spectrum_img
        #pin_wait_change('pin_select_spectrum')

    def render_spectra_table():

        spectra_edit_table = [['\u2612', 'name', 'm/z', 'formula', '']]
        for i, s in enumerate(spectra):
            selected = spectra_process[i]
            char_selected = '\u2610'
            if(selected):
                char_selected = '\u2612'
            name = s['params']['title']
            mol_form = s['params'].get('formula')
            use_formula =  mol_form is not None
            mz = s['params']['pepmass'][0]
            spectra_edit_table.append([
                char_selected,
                name,
                mz,
                mol_form or '',
                put_buttons([{'label': 'edit', 'value': f'edit_{i}'}], onclick = lambda v : edit_spectrum(v))
            ])
        return spectra_edit_table

    def edit_spectrum(v):
        m = re.match('edit_([0-9]+)', v)
        i = int(m.group(1))
        print(f"clicked {i}")
        s = spectra[i]
        name = s['params']['title']
        mol_form = s['params'].get('formula')
        use_formula =  mol_form is not None
        spectrum_img = plot_spectrum(i)
        # popup(spectra[i]['params']['title'],

        #     [
        #         put_image(spectrum_img),
        #         put_buttons(['close'],onclick=lambda _: close_popup())
        #     ]
        # )
        with use_scope('output', clear=True):
            put_markdown(f'## {name}')
            put_image(spectrum_img)
        spectra_edited = input_group('Settings', [
            checkbox(name,
                [{'label': 'process', 'value': 'process', 'selected': spectra_process[i]},
                {'label': 'use fixed formula', 'value': 'use_mf', 'selected': use_formula}],
                name= f'spectrum_options'),
            input('Formula', 
                value = mol_form or '',
                name = f'spectrum_mf')
        ])
        print(f"edited spectrum {i}")

        if('use_mf' in spectra_edited['spectrum_options']):
            spectra[i]['params']['formula'] = spectra_edited['spectrum_mf']
        else:
            spectra[i]['params'].pop('formula', None)
        spectra_process[i] = 'process' in spectra_edited['spectrum_options']

        with use_scope('output', clear = True):
            put_loading()
        spectra_table = render_spectra_table()
        with use_scope('output', clear=True):
            put_table(
                spectra_table
            )


    with use_scope('output', clear = True):
            put_loading()
    spectra_table = render_spectra_table()
    with use_scope('output', clear=True):
        put_markdown("## Spectra selection and configuration")
        put_table(
            spectra_table
        )

    actions('Continue to SIRIUS settings', ['proceed'])

    spectra_edit_inputs = []
    for i, s in enumerate(spectra):
        name = s['params']['title']
        mol_form = s['params'].get('formula')
        use_formula =  mol_form is not None
        spectra_edit_inputs.extend([
            checkbox(name,
                [{'label': 'process', 'value': 'process'},
                {'label': 'use fixed formula', 'value': 'use_mf', 'selected': use_formula}],
                name= f'spectrum_options_{i}'),
            input('Formula', 
                value = mol_form or '',
                name = f'spectrum_mf_{i}'),
        ])

    put_markdown("## SIRIUS settings")

    options_profile = [
        {'label': 'Orbitrap', 'value': '-p orbitrap'},
        {'label': 'Q-TOF', 'value': '-p qtof'},
        {'label': 'custom (specify in CLI options)', 'value': ''}
    ]

    clear('output')
    #with use_scope('input_scope', clear = True):
    sirius_options = input_group(
        "SIRIUS options",
        [
            select('SIRIUS profile', options_profile, name='profile'),
            checkbox('', [{'value': 'use_zodiac', 'label': 'Use ZODIAC'}], name = 'use_zodiac'),
            input('Custom CLI options for formula', name='cli')
        ]
    )

    use_zodiac = ''
    if 'use_zodiac' in sirius_options['use_zodiac']:
        use_zodiac = ' zodiac '
    sirius_cli = f"formula {sirius_options['profile']} {sirius_options['cli']} {use_zodiac} structure -d ALL_BUT_INSILICO"
    sirius_out = os.path.join(
        sc.config['eval_folder'],
        f"sirius-{sc.config['eval_id']}")

    with use_scope('output', clear = True):
            put_text("SIRIUS is processing")
            put_loading()

    sirius_run = subprocess.Popen([
        'sirius.sh',
        "--log=warning",
        f"-i {target_path}",
        f"-o {sirius_out}",
        sirius_cli
        ])

    clear("output")
    output_sirius_progess = output(put_text("SIRIUS started"))
    with use_scope('output'):
        put_text("Calculating trees:")
        put_processbar('prog_trees')
        put_text("Predicting fingerprints:")
        put_processbar('prog_fingerprints')
        put_row([
            output_sirius_progess,
        ])

    while(sirius_run.poll() is None):
        sirius_path = pathlib.Path(sirius_out)
        if sirius_path.exists():
            sirius_spectra = [x for x in sirius_path.iterdir() if x.is_dir()]
            sirius_trees_complete = sum([
                (x / "trees").exists() for x in sirius_spectra
            ])
            sirius_fp_complete = sum([
                (x / "fingerprints").exists() for x in sirius_spectra
            ])
            n_total = len(sirius_spectra)
            set_processbar('prog_trees', float(sirius_trees_complete)/float(n_total))
            set_processbar('prog_fingerprints', float(sirius_fp_complete)/float(n_total))
            
            #with use_scope('output'):
            output_sirius_progess.reset(
                put_text(f"total spectra: {n_total}, trees complete: {sirius_trees_complete}, fingerprints_complete: {sirius_fp_complete}")   
            )

        time.sleep(1)

    with use_scope('output', clear = True):
            put_text("SIRIUS is done")

    

    
    # put_select(
    #     name = 'pin_select_spectrum',
    #     label = "Choose spectrum",
    #     options = [
    #         {
    #             'label': name,
    #             'value': i,
    #             'selected': i == 0
    #         }
    #         for i, name in enumerate(spectra_names)
    #     ]
    # )

    # status = {'proceed': False }
    # def proceed():
    #     status['proceed'] = True
    #     pin_update('pin_select_spectrum')


    # while status['proceed'] == False:
    #     tf, spectrum_temp_png = tempfile.mkstemp(suffix = ".png")
    #     os.close(tf)
    #     spectrum_fig = plt.figure()
    #     spectrum_fig.add_subplot(sup.spectrum(spectra_vis[pin.pin_select_spectrum])) 
    #     spectrum_fig.savefig(spectrum_temp_png)
    #     spectrum_img = open(spectrum_temp_png, 'rb').read()
    #     with use_scope('shape', clear=True):
    #         put_text(spectra_names[pin.pin_select_spectrum])
    #         put_text(status['proceed'])
    #         put_image(spectrum_img)
    #         put_buttons([
    #             dict({
    #                 'label': 'proceed',
    #                 'value': 'proceed',
    #                 'color': 'success'
    #             })],
    #             onclick = lambda _ : proceed() )
    #     #pin_wait_change('pin_select_spectrum')


pywebio.start_server(main, port=int(8050), host = '0.0.0.0')
