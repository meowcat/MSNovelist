import smiles_config as sc
import subprocess
import shutil
import os

def run_sirius(mgf_path, formula = None, ion = "[M+H]+",
               precursor = None,
               out_id = None, profile='orbitrap'):
    if out_id is None:
        out_id = str(int(hash(mgf_path) % 1e8))
    out_folder = sc.config['base_folder'] + "sirius_out/" + out_id

    sirius_args = []
    sirius_cmd = [sc.config['sirius_bin']]
    sirius_args.extend(['-o', out_folder])
    sirius_args.extend(['-F'])
    if formula is not None:
        sirius_args.extend(['-f', formula])
    if ion is not None:
        sirius_args.extend(['-i', ion])
    sirius_args.extend(['--profile', profile])
    if precursor is not None:
        sirius_args.extend(['--mz', precursor])
        
    sirius_cmd.extend(sirius_args)
    sirius_cmd.append(mgf_path)
    
    capture = subprocess.run(
    args=sirius_cmd, capture_output=True)
    return out_id, capture


def retrieve_fingerprint(result_dir):
    dir_contents = os.scandir(result_dir, )
    subdir = [x for x in dir_contents if x.is_dir()]
    if len(subdir) < 1:
        return None
    subdir = subdir[0].path
    subdir = subdir + "/fingerprints"
    if not os.path.isdir(subdir):
        return None
    fingerprint = os.scandir(subdir)
    return next(fingerprint).path
    
    