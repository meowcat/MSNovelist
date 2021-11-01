# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:07:42 2021

@author: stravsm
"""

from pyteomics import mgf
import argparse

parser = argparse.ArgumentParser(
    description='Helper to split MGF files')
parser.add_argument('--chunk-size', '-c')
parser.add_argument('--files', '-f')
parser.add_argument('--prefix', '-p', required = False, default = "spectra_out_")
parser.add_argument('--prefix-title', '-t', action = 'store_true')
parser.add_argument('filename')

args = parser.parse_args()

mgf_file = args.filename
mgf_data = mgf.read(open(mgf_file, 'r'))
spectra = [s for s in mgf_data]


def update_title(i, s):
    title = s['params'].get('title', '')
    s['params'].update({'title': f'{i} {title}'})
    return s

if args.prefix_title:
    spectra = [update_title(i, s) for i, s in enumerate(spectra)]


spectra_len = len(spectra)

if args.chunk_size is not None:
    chunk_size = int(args.chunk_size)
elif args.files is not None:
    chunk_size = (spectra_len // int(args.files)) + 1
else:
    print("No chunk size or file number specified; splitting to individual files")
    chunk_size = 1

counter = 1

while len(spectra) > 0:
    chunk = spectra[:chunk_size]
    spectra = spectra[chunk_size:]
    mgf.write(chunk, f"{args.prefix}{counter}.mgf")
    counter = counter + 1
    
