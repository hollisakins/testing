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
from ..utils.filters import Filters
from .. import utils
from .. import config
from ..console import setup_logger




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

    def __init__(self, verbose=False):
        self._phot = []
        self._spec = []

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        pass
    
    @property
    def data(self):
        return {'phot':self.phot, 'spec':self.spec}
    
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


class Photometry(Observation):

    def __init__(self, 
                 filters: list | np.ndarray | brisket.filters.Filters, 
                 fluxes: list | np.ndarray | u.Quantity = None, 
                 errors: list | np.ndarray | u.Quantity = None, 
                 flux_unit: str | u.Unit = 'uJy'):
        
        self.filters = filters
        self.fluxes = fluxes
        self.errors = errors

        if not isinstance(self.filters, Filters):
            self.filters = Filters(self.filters)

        self.wav = self.filters.wav
        self.wav_min = self.filters.wav_min
        self.wav_max = self.filters.wav_max

        pass

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

    def __repr__(self):
        out = ''
        if self.fluxes is None:
            return self.filters.__repr__()
        if self.errors is None:
            for i in range(len(self)):
                out += f'{self.filters.nicknames[i]}: {self.fluxes[i]} \n'
        else:
            for i in range(len(self)):
                out += f'{self.filters.nicknames[i]}: {self.fluxes[i]} +/- {self.errors[i]} \n'
        return out
        # tree = Tree(f"[bold italic white]Params[/bold italic white](nparam={self.nparam}, ndim={self.ndim})")
        # self.filters.filter_nicknames
        # comps = list(self.components.keys())
        # names = [n for n in self.all_param_names if '/' not in n]
        # for name in names:
        #     tree.add('[bold #FFE4B5 not italic]' + name + '[white]: [italic not bold #c9b89b]' + self.all_params[name].__repr__())
        # for comp in comps:
        #     source = tree.add('[bold #6495ED not italic]' + comp + '[white]: [italic not bold #6480b3]' + self.components[comp].__repr__())#
        #     params_i = self.components[comp]
        #     names_i = [n for n in params_i.all_param_names if '/' not in n]
        #     for name_i in names_i:
        #         source.add('[bold #FFE4B5 not italic]' + name_i + '[white]: [italic not bold #c9b89b]' + params_i.all_params[name_i].__repr__())
        #     comps_i = list(params_i.components.keys())
        #     for comp_i in comps_i:
        #         subsource = source.add('[bold #8fbc8f not italic]' + comp_i + '[white]: [italic not bold #869e86]' + params_i.components[comp_i].__repr__())
        #         params_ii = params_i.components[comp_i]
        #         names_ii = [n for n in params_ii.all_param_names if '/' not in n]
        #         for name_ii in names_ii:
        #             subsource.add('[bold #FFE4B5 not italic]' + name_ii + '[white]: [italic not bold #c9b89b]' + params_ii.all_params[name_ii].__repr__())
        # console.print(tree)




class Spectrum(Observation):

    def __init__(self, 
                 wavs: list | np.ndarray | u.Quantity, 
                 fluxes: list | np.ndarray | u.Quantity = None,
                 errors: list | np.ndarray | u.Quantity = None,
                 mask: list | np.ndarray = None,
                 wav_units: str | u.Unit = None,
                 flux_units: str | u.Unit = None,
                 R: int = None):

        self.fluxes = fluxes
        self.errors = errors

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

        if self.fluxes is not None:
            # Remove points at the edges of the spectrum with zero flux.
            startn = 0
            while self.fluxes[startn] == 0.:
                startn += 1
            endn = 0
            while self.fluxes[-endn-1] == 0.:
                endn += 1
            if endn == 0:
                self.fluxes = self.fluxes[startn:]
                self.wavs = self.wavs[startn:]
            else:
                self.fluxes = self.fluxes[startn:-endn]
                self.wavs = self.wavs[startn:-endn]
        if self.errors is not None:
            if endn == 0:
                self.errors = self.errors[startn:]
            else:
                self.errors = self.errors[startn:-endn]

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
