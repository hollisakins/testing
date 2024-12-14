'''
Stellar models
'''
from __future__ import annotations

import numpy as np
import os
import h5py
import astropy.units as u

from .. import config
from ..utils import utils
from ..utils.sed import SED
# from ..data.grid_manager import GridManager
from ..grids.grids import Grid
from .base import *


class GriddedStellarModel(BaseGriddedModel, BaseSourceModel):
    '''
    The default stellar model in BRISKET. Handles input stellar grids 
    over a range of metallicities and ages.
    (TBE) Splits the resulting stellar SED into young and old components,
    to allow for e.g., extra dust attenuation in the young component.

    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''
    type = 'source'
    order = 0
    
    def __init__(self, params):
        self.params = params
        
        self._build_defaults(params)

        self.grid = Grid(str(params['grids']))
        self.grid.ages[self.grid.ages == 0] = 1
        self.grid.age_bins = np.power(10., utils.make_bins(np.log10(self.grid.ages), fix_low=-99))
        self.grid.age_widths = self.grid_age_bins[1:] - self.grid_age_bins[:-1]

    
    def build_defaults(self, params):
        if 'grids' not in params:
            params['grids'] = 'bc03_miles_chabrier'
        if 't_bc' not in params:
            params['t_bc'] = 0.01

    def validate_components(self, params):
        # TODO initialize SFHs
        # stack all the SFH components into one, there's no real point in keeping them separate
        self.sfh_components = {}
        for comp_name, comp in self.params.components.items():
            if comp.model.type == 'sfh':
                self.sfh_components[comp_name] = comp.model
        if len(self.sfh_components) == 1:
            self.sfh_weights = [1]
        else:
            if 'weight' in comp:
                self.sfh_weights = [float(comp['weight']) for comp in self.params.components.values() if comp.model.type == 'sfh']
            elif 'logweight' in comp:
                self.sfh_weights = [np.power(10., float(comp['logweight'])) for comp in self.params.components.values() if comp.model.type == 'sfh']

    def __repr__(self):
        return f'BaseStellarModel(grid={self.grid.name}, sfh={list(self.sfh_components)})'
    
    def __str__(self):
        return self.__repr__()

    def resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths
        self.grid.resample(self.wavelengths)

    def emit(self, params):
    
        """ Obtain a split 1D spectrum for a given star-formation and
        chemical enrichment history, one for ages lower than t_bc, one
        for ages higher than t_bc. This allows extra dust to be applied
        to the younger population still within its birth clouds.

        parameters
        ----------

        sfh_ceh : numpy.ndarray
            2D array containing the desired star-formation and
            chemical evolution history.

        t_bc : float
            The age at which to split the spectrum in Gyr.
        """

        # TODO compute sfh_ceh from input SFH parameters
        sfh_ceh_weights = np.zeros(grid.shape)
        for (sfh_name, sfh), sfh_weight in zip(self.sfh_components.items(), self.sfh_weights):
            sfh.update(params[sfh_name], weight=sfh_weight)
            sfh_ceh_weights += sfh.combined_weights

        # split the grid into young and old components at t_bc
        # (have to account for the grid age spacing)
        t_bc = float(params['t_bc']) * 1e9
        index = self.grid_ages[self.grid_ages < t_bc].shape[0]
        if index == 0:
            index += 1

        grid_young = self.grid[self.grid.age < t_bc]
        weight_young = (self.grid.age[index] - t_bc)/self.grid_age_widths[index-1]

        grid_old = self.grid[self.grid.age >= t_bc]
        weight_old = 1-weight_young

        sfh_ceh_young = copy(sfh_ceh[:, :index])
        sfh_ceh_young[:, index-1] *= weight_young
        grid_young.collapse(axis=('zmet','age'), weights=sfh_ceh_young, inplace=True)
        young = grid_young.to_SED()

        sfh_ceh_old = copy(sfh_ceh[:, index-1:])
        sfh_ceh_old[:, index-1] *= weight_old
        grid_old.collapse(axis=('zmet','age'), weights=sfh_ceh_old, inplace=True)
        old = grid_old.to_SED()

        # TODO: build some complexity into the SED class to handle separate young+old components, if specified 
        # but still handle units properly, with astropy equivalencies 
        return young+old

class SSPModel(BaseGriddedModel, BaseSourceModel):
    '''
    A simple stellar population model, which interpolates from 
    a given stellar grid to the specified age and metallicity (zmet).

    Args:
        params (brisket.parameters.Params)
            Model parameters.
    '''
    type = 'source'
    order = 0

    def __init__(self, params):
        self.params = params
        # self._build_defaults(params)
        self.grid = Grid(str(params['grids']))

    def validate_components(self, params):
        pass

    def resample(self, wavelengths):
        """ Resamples the raw stellar grids to the input wavs. """
        self.wavelengths = wavelengths
        self.grid.resample(self.wavelengths)

    def emit(self, params):
        self.grid.interpolate({'zmet':0, 'age':0}, inplace=True)
        # interpolate live_frac
        # scale to mass: self.grid *= params['massformed']
        
        return self.grid.to_SED()
