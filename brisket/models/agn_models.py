'''
AGN models
'''
from __future__ import annotations

import numpy as np

import astropy.units as u
from astropy.constants import c
c = c.to(u.angstrom*u.Hz).value

from .. import config
from ..utils.sed import SED
from .base_models import *
from ..console import setup_logger

class PowerlawAccrectionDiskModel(BaseFunctionalModel, BaseSourceModel):
    type = 'source'
    order = 1
    
    def __init__(self, params):
        self._build_defaults(params)
        super().__init__(params)

    def _build_defaults(self, params):
        if not 'beta' in params:
            params['beta'] = -2.0
        if not 'Muv' in params:
            raise Exception("Key Muv must be specified in parameters")

    def emit(self, params):
        beta, Muv, redshift = float(params['beta']), float(params['Muv']), float(params['redshift']) # absolute magnitude
        sed = SED(wav_rest=self.wavelengths, flam=np.power(self.wavelengths, beta), redshift=redshift, verbose=False)
        sed *= np.power(10., -0.4*(Muv-sed.Muv))
        return sed

