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

import sys
import os
sys.path.append(os.environ['MSNOVELIST_BASE'])

import smiles_config as sc


def main():
    put_markdown("# MSNovelist web interface")

    # Upload and store spectra
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

    def plot_spectrum(i):
        tf, spectrum_temp_png = tempfile.mkstemp(suffix = ".png")
        os.close(tf)
        spectrum_fig = plt.figure()
        spectrum_fig.add_subplot(sup.spectrum(spectra_vis[i])) 
        spectrum_fig.savefig(spectrum_temp_png)
        spectrum_img = open(spectrum_temp_png, 'rb').read()
        with use_scope('shape', clear=True):
            put_text(spectra_names[i])
            put_image(spectrum_img)
        #pin_wait_change('pin_select_spectrum')


    spectrum_sel = select(
        label = "Choose spectrum",
        options = [
            {
                'label': name,
                'value': i,
                'selected': i == 0
            }
            for i, name in enumerate(spectra_names)
        ],
        onchange = plot_spectrum
    )


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


    put_text("Done")


pywebio.start_server(main, port=int(8050), host = '0.0.0.0')
