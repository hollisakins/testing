"""
Module for handling observational data in BRISKET. 

The ``Observation`` class serves as the interface between idealized models generated with the 
``Model`` class and actual, observed data. It can be used to provide ``Model`` with the 
necessary info to generate proper synthetic observations (photometry or spectra) and to provide 
the ``Fitter`` class with real data to fit. 

BRISKET provides two subclasses of ``Observation``, ``Photometry`` and ``Spectrum`` for handling
photometry and spectroscopic data, respectively. 

.. highlight:: python

:: 

    obs = Observation()
    obs.add_phot(filters, fluxes, errors)
    obs.add_spec(wavs, fluxes, errors)

Generating a simple model SED does not require specifying any information about the observation. 
However, it will not yield any observables (photometry/spectra), only the internal model SED, and
will be generated at the configured default wavelength resolution (``config.R_default``), which 
may not be sufficient for predicting observables.  

::

    mod = ModelGalaxy(params) 

You can add an observation after the fact, which will determine the optimal wavelength sampling and 
re-sample the model 

::

    mod.add_obs(obs) # resamples to optimized wavelength resolution

However, it is often better to just provide the observation info from the initial ModelGalaxy construction: 

::

    mod = ModelGalaxy(params, obs=obs)

When fitting data, the Observation class becomes the mechanism for providing the data to fit. 

::

    obs = Photometry(filters, fluxes, errors) + Spectrum(wavs, fluxes, errors, calib)
    result = Fitter(params, obs).fit()
"""
from __future__ import annotations


import numpy as np
import sys, os
from rich.tree import Tree
import astropy.units as u
from .utils.filters import Filters
from . import utils
from . import config
from .console import setup_logger
from .utils import SED




# # or a more complicated observation
# import brisket
# phot_optical = brisket.Photometry(filters=filters, fluxes=fluxes, errors=errors)
# phot_fir = brisket.Photometry(filters=[brisket.filters.TopHat(850*u.micron, 1*u.micron)], fluxes=[...], errors=[...])
# spec_G235M = brisket.Spectrum(wavs, fluxes, errors, calibration='calib_g235m')
# spec_G395M = brisket.Spectrum(wavs, fluxes, errors, calibration='calib_g395m')
# observation = phot_optical + phot_fir + spec_G235M + spec_G395M



class Observation:
    """
    A container for observational data loaded into BRISKET.

    """

    def __init__(self, 
                 ID: str | int = None, 
                 verbose: bool = False):
        self.ID = ID
        self._phot = []
        self._spec = []

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        pass
    
    @property
    def phot_list(self):
        if isinstance(self, Spectrum):
            return None
        return self._phot
    
    @property
    def phot(self):
        if isinstance(self, Spectrum):
            return None
        if len(self._phot) == 1:
            return self._phot[0]
        else:
            return self._phot
    
    @property 
    def N_phot(self):
        return len(self._phot)
    
    def add_phot(self, *args, **kwargs):
        phot = Photometry(*args, **kwargs)
        self._phot.append(phot)
    
    @property
    def spec_list(self):
        if isinstance(self, Photometry):
            return None
        return self._spec

    @property
    def spec(self):
        if isinstance(self, Photometry):
            return None
        if len(self._spec) == 1:
            return self._spec[0]
        else:
            return self._spec

    @property 
    def N_spec(self):
        return len(self._spec)

    def add_spec(self, *args, **kwargs):
        spec = Spectrum(*args, **kwargs)
        self._spec.append(spec)


class Photometry(SED):

    def __init__(self, 
                 filters: list | np.ndarray | brisket.filters.Filters, 
                 fnu: list | np.ndarray | u.Quantity = None, 
                 fnu_err: list | np.ndarray | u.Quantity = None, 
                 flam: list | np.ndarray | u.Quantity = None, 
                 flam_err: list | np.ndarray | u.Quantity = None, 
                 fnu_units: str | u.Unit = 'uJy',
                 flam_units: str | u.Unit = 'ergscm2a', 
                 verbose: bool = False, **kwargs):
        
        self.filters = filters
        # self.fluxes = fluxes
        # self.errors = errors

        if not isinstance(self.filters, Filters):
            self.filters = Filters(self.filters)

        self.wav = self.filters.wav
        self.wav_min = self.filters.wav_min
        self.wav_max = self.filters.wav_max

        if fnu is not None:
            if not hasattr(fnu, 'unit'):
                if isinstance(fnu_units, str):
                    fnu_units = utils.unit_parser(fnu_units)
                fnu *= fnu_units
            if fnu_err is not None:
                fnu_err *= fnu_units
                args = {'filters':self.filters, 'fnu':fnu, 'fnu_err':fnu_err, 'verbose':verbose, 'units':True}
            else:
                args = {'filters':self.filters, 'fnu':fnu, 'verbose':verbose, 'units':True}
        elif flam is not None:
            if not hasattr(flam, 'unit'):
                if isinstance(flam_units, str):
                    flam_units = utils.unit_parser(flam_units)
                flam *= flam_units
            if flam_err is not None:
                flam_err *= fnu_units
                args = {'filters':self.filters, 'flam':flam, 'flam_err':flam_err, 'verbose':verbose, 'units':True}
            else:
                args = {'filters':self.filters, 'flam':flam, 'verbose':verbose, 'units':True}
        else:
            args = {'filters':self.filters, 'verbose':verbose, 'units':True}
        super().__init__(**args, **kwargs)

    @property
    def R(self):
        # compute pseudo-resolution based on the input filters 
        dwav = self.filters.wav_max - self.filters.wav_min
        return np.max(self.filters.wav/dwav)

    @property
    def wav_range(self):
        return np.min(self.filters.wav_min), np.max(self.filters.wav_max)
    
    def __len__(self):
        return len(self.filters)

    # def __repr__(self):
    #     out = ''
    #     return out



class Spectrum(Observation):

    def __init__(self, 
                 wavs: list | np.ndarray | u.Quantity, 
                 wav_units: str | u.Unit = None,
                 R: int = None,
                 **kwargs):
                #  flam: list | np.ndarray | u.Quantity = None,
                #  err: list | np.ndarray | u.Quantity = None,
                #  mask: list | np.ndarray = None,
                #  flux_units: str | u.Unit = None,
        
        _y_keys = ['Llam', 'Lnu', 'flam', 'fnu', 'nuLnu', 'lamLlam', 'nufnu', 'lamflam']
        _y_defs = [k in kwargs for k in _y_keys]
        if sum(_y_defs)==0:
            self.flux = None
            self.err = None
        elif sum(_y_defs)>1:
            self.logger.error("Must supply at most one specification spectral flux"); sys.exit()
        else:
            self._y_key = _y_keys[_y_defs.index(True)]
            self.flux = kwargs[self._y_key]
        
            _yerr_keys = ['err','error',self._y_key+'_err']
            _yerr_defs = [k in kwargs for k in _yerr_keys]
            if sum(_yerr_defs)==0:
                self.err = None
            elif sum(_yerr_defs)>1:
                self.logger.error("Must supply at most one specification spectral error"); sys.exit()
            else:
                _yerr_key = _yerr_keys[_yerr_defs.index(True)]
                self.err = kwargs[_yerr_key]

        if hasattr(wavs, 'unit'):
            self.wavs = wavs
            # self.wavs = wavs.to(u.angstrom).value
        elif wav_units is not None:
            if isinstance(wav_units, str):
                wav_units = utils.unit_parser(wav_units)
            self.wavs = (wavs*wav_units)
        else:
            self.logger.warning(f'Spectrum: no units provided for wavs, assuming {config.default_wavelength_unit}')
            self.wavs = (wavs*config.default_wavelength_unit)
    
        self._R = R
        # self.errors # apply mask

        if self.flux is not None:
            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.flux[startn] == 0.:
                startn += 1
            endn = 0
            while self.flux[-endn-1] == 0.:
                endn += 1
            if endn == 0:
                self.flux = self.flux[startn:]
                self.wavs = self.wavs[startn:]
            else:
                self.flux = self.flux[startn:-endn]
                self.wavs = self.wavs[startn:-endn]
        if self.err is not None:
            if endn == 0:
                self.err = self.err[startn:]
            else:
                self.err = self.err[startn:-endn]

    @property
    def R(self):
        if self._R is not None:
            return self._R
        # compute pseudo-resolution based on the input spec_wavs
        return np.max((self.wavs.value[1:]+self.wavs.value[:-1])/2/np.diff(self.wavs.value))


    def _mask(self, spec):
        """ Set the error spectrum to infinity in masked regions. """
        pass

    def __add__(self, other):
        result = Observation()
        # result.photometry += self

    @property
    def wav_range(self):
        return np.min(self.wavs.to(u.angstrom).value), np.max(self.wavs.to(u.angstrom).value)
