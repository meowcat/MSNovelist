from pyteomics import mgf
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus


def get_su_spectrum(spec):
    return sus.MsmsSpectrum(
        identifier = spec['params']['title'],
        mz = spec['m/z array'],
        intensity = spec['intensity array'],
        precursor_mz = spec['params']['pepmass'][0],
        precursor_charge = spec['params']['charge'][0]
    )

