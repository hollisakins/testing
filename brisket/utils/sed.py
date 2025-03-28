"""
SED module for handling galaxy spectral energy distributions.


Example usage:

.. highlight:: python

::

    from brisket.utils.sed import SED
    sed = SED(wav_rest = ..., total = ..., redshift=5, frame='observed')

    sed = sed.to(xunit=u.angstrom, yunit=u.erg/u.s/u.cm**2/u.angstrom)

    ax.plot(sed['wav_rest'], sed['total']


"""
from __future__ import annotations
from collections.abc import Iterable

import sys
import numpy as np  
from copy import deepcopy
from astropy.units import Unit, Quantity, spectral_density, UnitTypeError
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant
import astropy.units as u
import spectres

from .. import config
from . import utils
from .filters import Filters
border_chars = config.border_chars
from brisket.console import setup_logger, rich_str
from rich.tree import Tree

import matplotlib.pyplot as plt
import matplotlib as mpl





np_handled_array_functions = {}


class SED:
    '''
    Primary class for manipulating galaxy SEDs.

    Args:
        wav_rest (array-like)
            The rest-frame wavelengths of the SED. 
        redshift (float)
            Redshift of the source (defaults to 0).
        verbose (bool)
            Verbosity flag (defaults to True).
        **kwargs
            Flux specification: no more than one of 'Llam', 'Lnu', 'flam', 'fnu', 'nuLnu', 'lamLlam', 'nufnu', 'lamflam' 
            If none are provided, the SED will be populated with zeros (in L_lam).

    
    Attributes:
        wav_rest (array-like)
            The rest-frame wavelengths of the SED. 

    '''
    
    def __init__(self, 
                 redshift: float | None = None, 
                 verbose: bool = True, 
                 # x-axis specification (only one!)
                 wav_rest: Iterable[float] = None, 
                 wav_obs: Iterable[float] = None, 
                 freq_rest: Iterable[float] = None, 
                 freq_obs: Iterable[float] = None, 
                 energy_rest: Iterable[float] = None, 
                 energy_obs: Iterable[float] = None, 
                 filters: Iterable[str] | Filters = None, 
                 # y-axis specification (can have multiple)
                 total: Iterable[float] = None,
                 **components):

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')
        
        self.redshift = redshift
        

        ### x-axis specification
        self._x_keys = ['wav_rest', 'wav_obs', 'freq_rest', 'freq_obs', 'energy_rest', 'energy_obs', 'filters']
        self._x = [wav_rest, wav_obs, freq_rest, freq_obs, energy_rest, energy_obs, filters]
        self._x_defs = [k is not None for k in self._x]
        if sum(self._x_defs) != 1:
            self.logger.error(f"""Must supply exactly one specification of the SED 
                                  wavelength/frequency/energy axis. 
                                  Options are: {self._x_keys}""")
            sys.exit()

        self._x_key = self._x_keys[self._x_defs.index(True)]
        self._x = self._x[self._x_defs.index(True)]
        

        if self._x_key == 'filters':
            if isinstance(self._x, Filters):
                self.filters = self._x
            elif isinstance(self._x, (list,np.ndarray)):
                self.filters = Filters(self._x)
            else:
                raise Exception
            #TODO make sure that the filter wavelengths have units, IF y is provided with units
            self._x = self.filters.wav
            self._x_key = 'wav_obs'
            self.units = True
        else:
            self.filters = None
            self.units = hasattr(self._x, "unit")
            
        ### y-axis specification
        self._y = components
        if total is None:
            self.logger.debug('No flux/luminosity information provided, populating with zeros. If this is intended, you can ignore this message.')
            self._y['total'] = np.zeros(len(self._x))
            self.units = False
        else:
            self._y['total'] = total
            self.units = self.units and hasattr(total, "unit")

        # perform some checks
        assert all([hasattr(c, "__len__") for c in self._y.values()]), "All SED components must be iterable."
        assert len(set([len(c) for c in self._y.values()])) == 1, "All SED components must have the same length."
        if self.units:
            assert len(set([self._check_y_type(c.unit) for c in self._y.values()])) == 1, "All SED components must have the same physical type."
            assert len(set([c.unit for c in self._y.values()])) == 1, "All SED components must have the same units."

        if self.filters is not None and self.units: 
            pass
            # TODO add units to filters
            # self._x *= self.filters.wav.unit


        # if units:
            # x_default_units = {'wav_rest':config.default_wavelength_unit, 
            #                    'wav_obs':config.default_wavelength_unit,
            #                    'freq_rest':config.default_frequency_unit,
            #                    'freq_obs':config.default_frequency_unit,
            #                    'energy_rest':config.default_energy_unit,
            #                    'energy_obs':config.default_energy_unit}
            # if not hasattr(self._x, "unit"):
            #     self.logger.info(f"No units specified for {self._x_key}, adopting default ({x_default_units[self._x_key]})")
            #     self._x *= x_default_units[self._x_key]

            # y_default_units = {'Llam': config.default_Llam_unit, 
            #                    'Lnu': config.default_Lnu_unit, 
            #                    'flam': config.default_flam_unit, 
            #                    'fnu': config.default_fnu_unit, 
            #                    'nuLnu': config.default_lum_unit, 
            #                    'lamLlam': config.default_lum_unit, 
            #                    'nufnu': config.default_flux_unit, 
            #                    'lamflam': config.default_flux_unit}

            # if not hasattr(self._y, "unit"):
            #     self.logger.info(f"No units specified for {self._y_key}, adopting default ({y_default_units[self._y_key]})")
            #     self._y *= y_default_units[self._y_key]

            # if not hasattr(self._yerr, "unit"):
            #     self.logger.info(f"No units specified for {self._yerr_key}, adopting default ({y_default_units[self._y_key]})")
            #     self._yerr *= y_default_units[self._y_key]

            # # TODO handle case where units are provided for y but not yerr

            # self._x_unit = self._x.unit
            # self._y_unit = self._y.unit

    # def __getitem__(self, indices):
    #     '''Allows access to the flux array via direct indexing of the SED object'''
    #     newobj = deepcopy(self)
    #     for k in self.components.keys():
    #         newobj.components[k] = newobj.components[k][indices]
    #     return newobj
    
    # def __setitem__(self, indices, values):
    #     '''Allows setting of the flux array via direct indexing of the SED object'''
    #     self._y[indices] = values

    def __getitem__(self, key):
        if key in self._x_keys:
            return getattr(self, key)
        return self._y[key]
    
    def __setitem__(self, key, value):
        '''Allows setting of the flux array via direct indexing of the SED object'''
        # perform some checks
        assert np.shape(value) == np.shape(self._y['total']), "All SED components must have the same length."
        if self.units:
            assert self._check_y_type(value.unit) == self._y_type, "All SED components must have the same physical type."
            value = value.to(self._y_unit)
        self._y[key] = value



    #### x axis specification ##############################################################################################
    # the following methods handle the various ways to specify the x-axis values/units, and conversions between

    @property
    def _x_implemented(self):
        redshift = self.redshift is not None
        implemented = [self._x_key]
        if self._x_key=='wav_rest': 
            if self.units:
                implemented.append('freq_rest')
                implemented.append('energy_rest')
                if redshift:
                    implemented.append('wav_obs')
                    implemented.append('freq_obs')
                    implemented.append('energy_obs')
            else:
                if redshift:
                    implemented.append('wav_obs')
        elif self._x_key=='wav_obs': 
            if self.units:
                implemented.append('freq_obs')
                implemented.append('energy_obs')
                if redshift:
                    implemented.append('wav_rest')
                    implemented.append('freq_rest')
                    implemented.append('energy_rest')
            else:
                if redshift:
                    implemented.append('wav_rest')
        elif self._x_key=='freq_rest': 
            if self.units:
                implemented.append('wav_rest')
                implemented.append('energy_rest')
                if redshift:
                    implemented.append('wav_obs')
                    implemented.append('freq_obs')
                    implemented.append('energy_obs')
            else:
                if redshift:
                    implemented.append('freq_obs')
        elif self._x_key=='freq_obs': 
            if self.units:
                implemented.append('wav_obs')
                implemented.append('energy_obs')
                if redshift:
                    implemented.append('wav_rest')
                    implemented.append('freq_rest')
                    implemented.append('energy_rest')
            else:
                if redshift:
                    implemented.append('freq_rest')
        
        elif self._x_key=='energy_rest': 
            if self.units:
                implemented.append('wav_rest')
                implemented.append('freq_rest')
                if redshift:
                    implemented.append('wav_obs')
                    implemented.append('freq_obs')
                    implemented.append('energy_obs')
            else:
                if redshift:
                    implemented.append('energy_obs')
        elif self._x_key=='energy_obs': 
            if self.units:
                implemented.append('wav_obs')
                implemented.append('freq_obs')
                if redshift:
                    implemented.append('wav_rest')
                    implemented.append('freq_rest')
                    implemented.append('energy_rest')
            else:
                if redshift:
                    implemented.append('energy_rest')

        return implemented


    @property 
    def wav_rest(self) -> Quantity | np.ndarray:
        '''Rest-frame wavelengths'''
        if 'wav_rest' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest': 
            return self._x
        elif self._x_key=='wav_obs': 
            return self._x / (1+self.redshift)
        elif self._x_key=='freq_rest': 
            return (speed_of_light/self._x).to(config.default_wavelength_unit)
        elif self._x_key=='freq_obs': 
            return (speed_of_light/(self._x/(1+self.redshift))).to(config.default_wavelength_unit)
        elif self._x_key=='energy_rest': 
            return (plancks_constant * speed_of_light / self._x).to(config.default_wavelength_unit)
        elif self._x_key=='energy_obs': 
            return (plancks_constant * speed_of_light / self._x / (1+self.redshift)).to(config.default_wavelength_unit)

    @property 
    def wav_obs(self) -> Quantity | np.ndarray:
        '''Observed-frame wavelengths'''
        if 'wav_obs' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest': 
            return self._x * (1+self.redshift)
        elif self._x_key=='wav_obs': 
            return self._x
        elif self._x_key=='freq_rest': 
            return (speed_of_light/(self._x*(self.redshift))).to(config.default_wavelength_unit)
        elif self._x_key=='freq_obs': 
            return (speed_of_light/self._x).to(config.default_wavelength_unit)
        elif self._x_key=='energy_rest': 
            return (plancks_constant * speed_of_light / self._x * (1+self.redshift)).to(config.default_wavelength_unit)
        elif self._x_key=='energy_obs': 
            return (plancks_constant * speed_of_light / self._x).to(config.default_wavelength_unit)
            

    @property 
    def freq_rest(self) -> Quantity | np.ndarray:
        '''Rest-frame frequencies'''
        if 'freq_rest' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest':
            return (speed_of_light/self._x).to(config.default_frequency_unit)
        elif self._x_key=='wav_obs':
            pass
        elif self._x_key=='freq_rest':
            return self._x
        elif self._x_key=='freq_obs':
            return self._x * (1+self.redshift)
        elif self._x_key=='energy_rest':
            pass
        elif self._x_key=='energy_obs':
            pass

    @property 
    def freq_obs(self) -> Quantity | np.ndarray:
        '''Observed-frame frequencies'''
        if 'freq_obs' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest':
            pass
        elif self._x_key=='wav_obs':
            pass
        elif self._x_key=='freq_rest':
            return self._x / (1+self.redshift)
        elif self._x_key=='freq_obs':
            return self._x
        elif self._x_key=='energy_rest':
            pass
        elif self._x_key=='energy_obs':
            pass

    @property 
    def energy_rest(self) -> Quantity | np.ndarray:
        '''Rest-frame energies'''
        if 'energy_rest' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest':
            pass
        elif self._x_key=='wav_obs':
            pass
        elif self._x_key=='freq_rest':
            return (plancks_constant * self._x).to(config.default_energy_unit)
        elif self._x_key=='freq_obs':
            return (plancks_constant * self._x / (1+self.redshift)).to(config.default_energy_unit)
        elif self._x_key=='energy_rest':
            return self._x
        elif self._x_key=='energy_obs':
            return self._x * (1+self.redshift)
    
    @property 
    def energy_obs(self) -> Quantity | np.ndarray:
        '''Observed-frame energies'''
        if 'energy_obs' not in self._x_implemented:
            return NotImplemented
        if self._x_key=='wav_rest':
            pass
        elif self._x_key=='wav_obs':
            pass
        elif self._x_key=='freq_rest':
            pass
        elif self._x_key=='freq_obs':
            pass
        elif self._x_key=='energy_rest':
            return self._x / (1+self.redshift)
        elif self._x_key=='energy_obs':
            return self._x

    @property
    def _x_unit(self):
        return self._x.unit

    #### y axis specification ##############################################################################################
    # the following methods handle the various ways to specify the y-axis values/units, and conversions between

    @property
    def _y_unit(self):
        return self._y['total'].unit

    def _check_y_type(self, y_unit):
        if 'spectral flux density' in y_unit.physical_type: return 'fnu'
        elif 'spectral flux density wav' in y_unit.physical_type: return 'flam'
        elif 'energy flux' in y_unit.physical_type: return 'f'
        elif 'energy' in y_unit.physical_type: return 'Lnu'
        elif 'yank' in y_unit.physical_type: return 'Llam'
        elif 'power' in y_unit.physical_type: return 'L'
        else: raise UnitTypeError(f"Couldn't figure out the physical type of the current y-axis unit ({current_y_unit}).")

    @property 
    def _y_type(self):
        return self._check_y_type(self._y_unit)

    def convert_units(self, 
                      xunit: Unit = None, 
                      yunit: Unit = None, 
                      inplace: bool = True):

        if not self.units:
            self.logger.error("Cannot convert units for SED object without units. Assign units using SED.assign_units() first.")
            sys.exit()

        if inplace:
            sed = self
        else:
            sed = deepcopy(self)

        if xunit is not None:
            if isinstance(xunit, str):
                xunit = utils.unit_parser(xunit)
            # Convert x-units, and set _x to the new physical type / unit
            x_frame = sed._x_key.split('_')[1]
            if 'length' in xunit.physical_type:
                x_key = f'wav_{x_frame}'
            elif 'frequency' in xunit.physical_type:
                x_key = f'freq_{x_frame}'
            elif 'energy' in xunit.physical_type:
                x_key = f'energy_{x_frame}'
            sed._x = getattr(sed, _x_key).to(xunit)
            sed._x_key = x_key

        if yunit is not None:
            if isinstance(yunit, str):
                yunit = utils.unit_parser(yunit)
            fourPiLumDistSq = sed.fourPiLumDistSq
            
            to_fnu = 'spectral flux density' in yunit.physical_type
            to_flam = 'spectral flux density wav' in yunit.physical_type
            to_f = 'energy flux' in yunit.physical_type
            to_Lnu = 'energy' in yunit.physical_type
            to_Llam = 'yank' in yunit.physical_type
            to_L = 'power' in yunit.physical_type
            current = sed._y_type

            # not changing physical type (i.e., fnu->fnu, flam->flam, etc.)
            if (current == 'f' and to_f) or (current == 'L' and to_L) or (current == 'fnu' and to_fnu) or (current == 'flam' and to_flam) or (current == 'Lnu' and to_Lnu) or (current == 'Llam' and to_Llam):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y).to(yunit) 
            
            # going from fnu->flam, flam->fnu, Lnu->Llam, Llam->Lnu
            # TODO how does equivalencies handle rest-frame vs observed-frame?
            elif (current == 'fnu' and to_flam) or (current == 'flam' and to_fnu) or (current == 'Lnu' and to_Llam) or (current == 'Llam' and to_Lnu): # easy to convert between fnu and flam
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = component_y.to(yunit, equivalencies=spectral_density(sed.wav_obs)) 

            # going from fnu->Lnu, flam->Llam, or f->L
            elif (current == 'fnu' and to_Lnu) or (current == 'flam' and to_Llam) or (current == 'f' and to_L):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq).to(yunit) 
            
            # going from Lnu->fnu, Llam->flam, or L->f
            elif (current == 'Lnu' and to_fnu) or (current == 'Llam' and to_flam) or (current == 'L' and to_f):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq).to(yunit) 

            # going from fnu->Llam or flam->Lnu
            elif (current == 'fnu' and to_Llam) or (current == 'flam' and to_Lnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq).to(yunit, equivalencies=spectral_density(sed.wav_obs)) 
            
            # going from Lnu->flam or Llam->fnu
            elif (current == 'Lnu' and to_flam) or (current == 'Llam' and to_fnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq).to(yunit, equivalencies=spectral_density(sed.wav_obs)) 
            
            # specific implementations for converting to/from f
            elif (current == 'f' and to_fnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'f' and to_flam):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / sed.wav_obs).to(yunit)
            elif (current == 'f' and to_Lnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq / sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'f' and to_Llam):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq / sed.wav_obs).to(yunit)
            elif (current == 'fnu' and to_f):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'flam' and to_f):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * sed.wav_obs).to(yunit)
            elif (current == 'Lnu' and to_f):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq * sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'Llam' and to_f):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq * sed.wav_obs).to(yunit)
            
            # specific implementations for converting to/from L
            elif (current == 'L' and to_fnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq / sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'L' and to_flam):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / fourPiLumDistSq / sed.wav_obs).to(yunit)
            elif (current == 'L' and to_Lnu):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'L' and to_Llam):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y / sed.wav_obs).to(yunit)
            elif (current == 'fnu' and to_L):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq * sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'flam' and to_L):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * fourPiLumDistSq * sed.wav_obs).to(yunit)
            elif (current == 'Lnu' and to_L):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * sed.freq_obs).to(yunit) # TODO handle frame
            elif (current == 'Llam' and to_L):
                for component_name, component_y in sed._y.items():
                    sed._y[component_name] = (component_y * sed.wav_obs).to(yunit)

            else:
                self.logger.error(f"Couldn't figure out how to convert from {current_y_unit} to {yunit}. Perhaps this is not implemented yet?")
                sys.exit()
            
            if not inplace:
                return sed
        
        
        

    def assign_units(self, xunit: Unit, yunit: Unit, inplace=True) -> None:
        '''
        Assigns units to the SED object. Used when the SED is constructed without units (for speed) but 
        units are needed for further calculations. 
        '''
        if inplace:
            self._x *= xunit
            for k in self._y.keys():
                self._y[k] *= yunit
            self.units = True
            return
        else:
            new = deepcopy(self)
            new._x *= xunit
            for k in new._y.keys():
                new._y[k] *= yunit
            new.units = True
            return new


    # @property
    # def fnu(self):
    #     '''
    #     Spectral flux density in terms of flux per unit frequency. Automatically converts from the flux specification defined at construction.
    #     '''
    #     if self._y_key=='fnu':
    #         return (self._y).to(config.default_fnu_unit)
    #     elif self._y_key=='flam':
    #         return (self._y * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
    #     elif self._y_key=='Lnu':
    #         return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
    #     elif self._y_key=='Llam':
    #         return (self._y / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
    #     elif self._y_key=='L':
    #         return (self._y / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
    #     elif self._y_key=='f':
    #         return (self._y / self.nu_obs).to(config.default_fnu_unit)
    #     else:
    #         raise Exception

    
    # @property
    # def flam(self):
    #     '''
    #     Spectral flux density in terms of flux per unit wavelength. Automatically converts from the flux specification defined at construction.
    #     '''
    #     if self._y_key=='fnu':
    #         return (self._y / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
    #     elif self._y_key=='flam':
    #         return (self._y).to(config.default_flam_unit)
    #     elif self._y_key=='Lnu':
    #         return (self._y / (4*np.pi*self.luminosity_distance**2) / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
    #     elif self._y_key=='Llam':
    #         return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
    #     elif self._y_key=='L':
    #         return (self._y / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
    #     elif self._y_key=='f':
    #         return (self._y / self.lam_obs).to(config.default_flam_unit)
    #     else:
    #         raise Exception
    
    # @property
    # def Llam(self):
    #     '''
    #     Spectral flux density in terms of flux per unit wavelength. Automatically converts from the flux specification defined at construction.
    #     '''
    #     if self._y_key=='fnu':
    #         pass
    #     elif self._y_key=='flam':
    #         pass # return (self._y).to(config.default_flam_unit)
    #     elif self._y_key=='Lnu':
    #         return (self._y / self.wav_obs**2 * speed_of_light).to(config.default_Llam_unit)
    #     elif self._y_key=='Llam':
    #         return self._y
    #         # return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
    #     elif self._y_key=='L':
    #         pass # return (self._y / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
    #     elif self._y_key=='f':
    #         pass # return (self._y / self.lam_obs).to(config.default_flam_unit)
    #     else:
    #         raise Exception


    #################################################################################
    def resample(self, fill=0, **kwargs):
        _x_defs = [k in kwargs for k in self._x_keys]
        if sum(self._x_defs) != 1:
            self.logger.error(f"""Resample: Must supply exactly one specification 
                                  of the wavelength/frequency/energy axis. 
                                  Options are: {self._x_keys}""")
            sys.exit()

        _new_x_key = self._x_keys[_x_defs.index(True)]
        _new_x = kwargs.get(_new_x_key)
        _old_x = getattr(self, _new_x_key)
    
        if self.units:
            self._y = spectres.spectres(_new_x.value, _old_x.to(_new_x.unit).value, self._y.value, fill=fill, verbose=False) * self._y.unit
        else:
            self._y = spectres.spectres(_new_x, _old_x, self._y, fill=fill, verbose=False)
        
        self._x = _new_x
        self._x_key = _new_x_key
        # quit()
        return self._y

    def __repr__(self):

        tree = Tree(f"[bold italic white]BRISKET-SED[/bold italic white](ndim={self.ndim}, redshift={self.redshift}, units={self.units})")
        # x-axis: {self._x_key} = {self._x}
        # available (computed on the fly): {*self._x_keys}
        # can convert to: 

        # y-axis: total ({self._y_type}) = {self.total} {self._y_unit}
        #         {name} ({self._y_type}) = {self.name} {self._y_unit}
        # can convert to: 

        if self.units:
            if self.filters is not None:
                x = tree.add('[bold #6495ED not italic]x-axis[white]: [italic not bold]' + f'[green]filters[white not italic]: {", ".join(self.filters.nicknames)} [dark_red]{np.shape(self._x)}')
            else:
                x = tree.add('[bold #6495ED not italic]x-axis[white]: [italic not bold]' + f'[green]{self._x_key}[white not italic]: [{self._x.value[0]:.2f}, {self._x.value[1]:.2f}, ..., {self._x.value[-2]:.2f}, {self._x.value[-1]:.2f}] [green]{self._x_unit} [dark_red]{np.shape(self._x)}')
            
            impl = [f for f in self._x_implemented if f != self._x_key]
            x.add(f'available: [dark_green italic]{", ".join(impl)}')

            if self.ndim==1:
                y = tree.add('[bold #6495ED not italic]y-axis[white]: [italic not bold]' + f'[green]total[white not italic]: [{self._y['total'].value[0]:.2f}, {self._y['total'].value[1]:.2f}, ..., {self._y['total'].value[-2]:.2f}, {self._y['total'].value[-1]:.2f}] [green]{self._y_unit} [dark_red]{np.shape(self._y['total'])}')
                for name, value in self._y.items():
                    if name == 'total': continue
                    y.add(f'[green italic]{name}[white not italic]: [{value.value[0]:.2f}, {value.value[1]:.2f}, ..., {value.value[-2]:.2f}, {value.value[-1]:.2f}] [green]{self._y_unit} [dark_red]{np.shape(value)}')
            else:
                y = tree.add('[bold #6495ED not italic]y-axis[white]: [italic not bold]' + f'[green]total[white not italic]: [...] [green]{self._y_unit} [dark_red]{np.shape(self._y['total'])}')
                for name, value in self._y.items():
                    if name == 'total': continue
                    y.add(f'[green italic]{name}[white not italic]: [...] [green]{self._y_unit} [dark_red]{np.shape(value)}')
            
            props = tree.add('[bold #6495ED not italic]properties[white]:')
            if self.ndim == 1:
                props.add(f'[green]beta[white]: {self.properties['beta']:.3f}')
                props.add(f'[green]Muv[white]: {self.properties['Muv']:.3f}')
                props.add(f'[green]Lbol[white]: ?')
            elif self.ndim == 2:
                props.add(f'[green]beta[white]: [{self.properties['beta'][0]:.3f}, {self.properties['beta'][1]:.3f}, ..., {self.properties['beta'][-2]:.3f}, {self.properties['beta'][-1]:.3f}]')
                props.add(f'[green]Muv[white]: [{self.properties['Muv'][0]:.3f}, {self.properties['Muv'][1]:.3f}, ..., {self.properties['Muv'][-2]:.3f}, {self.properties['Muv'][-1]:.3f}]')
                props.add(f'[green]Lbol[white]: ?')
        else:
            if self.filters is not None:
                x = tree.add('[bold #6495ED not italic]x-axis[white]: [italic not bold]' + f'[green]filters[white not italic]: {", ".join(self.filters.nicknames)} [dark_red]{np.shape(self._x)}')
            else:
                x = tree.add('[bold #6495ED not italic]x-axis[white]: [italic not bold]' + f'[green]{self._x_key}[white not italic]: [{self._x[0]:.2f}, {self._x[1]:.2f}, ..., {self._x[-2]:.2f}, {self._x[-1]:.2f}] [dark_red]{np.shape(self._x)}')
            
            impl = [f for f in self._x_implemented if f != self._x_key]
            x.add(f'available: [dark_green italic]{", ".join(impl)}')

            if self.ndim==1:
                y = tree.add('[bold #6495ED not italic]y-axis[white]: [italic not bold]' + f'[green]total[white not italic]: [{self._y['total'][0]:.2f}, {self._y['total'][1]:.2f}, ..., {self._y['total'][-2]:.2f}, {self._y['total'][-1]:.2f}] [dark_red]{np.shape(self._y['total'])}')
                for name, value in self._y.items():
                    if name == 'total': continue
                    y.add(f'[green italic]{name}[white not italic]: [{value[0]:.2f}, {value[1]:.2f}, ..., {value[-2]:.2f}, {value[-1]:.2f}] [dark_red]{np.shape(value)}')
            
            else:
                y = tree.add('[bold #6495ED not italic]y-axis[white]: [italic not bold]' + f'[green]total[white not italic]: [...] [dark_red]{np.shape(self._y['total'])}')
                for name, value in self._y.items():
                    if name == 'total': continue
                    y.add(f'[green italic]{name}[white not italic]: [...] [dark_red]{np.shape(value)}')
            
            # props = tree.add('[bold #6495ED not italic]properties[white]:')
            # props.add(f'[green]beta[white]: ?')
            # props.add(f'[green]Muv[white]: ?')
            # props.add(f'[green]Lbol[white]: ?')


        # width = np.max([width, len(wstr)+4])
        # border_chars = '═║╔╦╗╠╬╣╚╩╝'
        # outstr = border_chars[2] + border_chars[0]*width + border_chars[4]
        # outstr += '\n' + border_chars[1] + 'BRISKET-SED'.center(width) + border_chars[1]
        # outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        # outstr += '\n' + border_chars[1] + wstr.center(width) + border_chars[1]
        # outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        # outstr += '\n' + border_chars[1] + fstr1.center(width) + border_chars[1]
        # outstr += '\n' + border_chars[1] + fstr2.center(width) + border_chars[1]
        # outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        # outstr += '\n' + border_chars[1] + betastr.center(width) + border_chars[1]
        # outstr += '\n' + border_chars[8] + border_chars[0]*width + border_chars[10]
        # return outstr
        # if np.ndim(f) == 1:
            # if len(self.wav_rest) > 4:
            #     wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
            #     fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnUnit}'
            # l = max((len(wstr),len(fstr)))
            # wstr = wstr.ljust(l+3)
            # fstr = fstr.ljust(l+3)
            # wstr += str(np.shape(w))
            # fstr += str(np.shape(f))

            # return f'''BRISKET-SED: wav: {wstr}, flux {np.shape(f)}'''
        # elif np.ndim(f)==2:
        #     if len(self.wav_rest) > 4:
        #         wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
        #         fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnUnit}'
        #     l = max((len(wstr),len(fstr)))
        #     wstr = wstr.ljust(l+3)
        #     fstr = fstr.ljust(l+3)
        #     wstr += str(np.shape(w))
        #     fstr += str(np.shape(f))

        #     return f'''BRISKET-SED: wav: {wstr}\n             fnu: {fstr}'''
        return '\n' + rich_str(tree)

    @property
    def ndim(self):
        return np.ndim(self._y['total'])

    def __str__(self):
        return self.__repr__()

    def __add__(self, other: SED) -> SED:
        '''
        Adds two SED objects together. 
        y-components present in both objects are summed together, y-components present in only one object are appended to the new object. 
        If the SED objects have different x-axis values, resamples the second SED to match the first.

        Ex: 
            # sed1 = SED(wav_rest=wav_rest, galaxy=...)
            # sed2 = SED(wav_rest=wav_rest, agn=...)
            # SED.from_several({'galaxy':sed1, 'agn':sed2})
        '''

        if not np.all(other._x==self._x):
            other.resample(**{self._x_key:self._x})
        
        newobj = SED(**{self._x_key:self._x}, redshift=self.redshift, verbose=False)
        all_components = list(set(self._y.keys()).union(set(other._y.keys())))
        for component_name in all_components:
            if component_name in self._y.keys() and component_name in other._y.keys():
                newobj._y[component_name] = self._y[component_name] + other._y[component_name]
            elif component_name in self._y.keys():
                newobj._y[component_name] = self._y[component_name]
            elif component_name in other._y.keys():
                newobj._y[component_name] = other._y[component_name]

        return newobj
    
    def __mul__(self, other: int | float | np.ndarray) -> SED: 
        newobj = deepcopy(self)
        for component_name, component_y in self._y.items():
            newobj._y[component_name] = component_y * other
        return newobj

    def __array__(self, dtype=None, copy=None):
        return self._y
    
    def __array_function__(self, func, types, args, kwargs):
        if func not in np_handled_array_functions:
            return NotImplemented
        # Allow subclasses (that don't override __array_function__) to handle SED objects
        # if not all(issubclass(t, self.__class__) for t in types):
        #     print('call __array_function__')
        #     return NotImplemented
        return np_handled_array_functions[func](*args, **kwargs)

    # def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    #     if method == '__call__':
    #         print('?')
    #         # newobj = SED(self.wav_rest, redshift=self.redshift, verbose=False)
    #         # args = [np.array(x._y) if isinstance(x, SED) else x for x in inputs]
    #         # setattr(newobj, '_y', ufunc(*args, **kwargs))
    #     else:
    #         return NotImplemented


        

    def measure_window_luminosity(self, window):
        return NotImplemented

    def measure_monochromatic_luminosity(self):
        return NotImplemented

    def measure_slope(self):
        return NotImplemented

    ########################################################################################################################
    @property 
    def luminosity_distance(self):
        if self.redshift is None:
            return NotImplemented
    
        if self.redshift == 0:
            dL =  10 * Unit('pc')
        else:
            dL = config.cosmo.luminosity_distance(self.redshift).to(Unit('pc'))
        if not self.units:
            dL = dL.value
        return dL
        
    @property
    def fourPiLumDistSq(self):
        return 4*np.pi*self.luminosity_distance**2

    @property
    def Lbol(self) -> Quantity:
        '''
        Bolometric luminosity of the SED.
        To be implemented.
        '''
        if self.ndim == 1:
            return np.nan
        elif self.ndim == 2:
            return np.nan * np.zeros(len(self._y['total']))

    @property
    def beta(self) -> float | np.ndarray:
        '''
        UV spectral slope measured using the Calzetti et al. (1994) spectral windows.
        If the SED has no unit information, returns NotImplemented.
        '''
        if not self.units:
            return NotImplemented
        w = self.wav_rest.to(Unit('angstrom')).value
        windows = ((w>=1268)&(w<=1284))|((w>=1309)&(w<=1316))|((w>=1342)&(w<=1371))|((w>=1407)&(w<=1515))|((w>=1562)&(w<=1583))|((w>=1677)&(w<=1740))|((w>=1760)&(w<=1833))|((w>=1866)&(w<=1890))|((w>=1930)&(w<=1950))|((w>=2400)&(w<=2580))
        if self._y_type != 'flam':
            s = self.convert_units(yunit='ergscm2a', inplace=False)
            y = s._y['total'].value
        else:
            y = self._y['total'].value
        if self.ndim == 1:
            p = np.polyfit(np.log10(w[windows]), np.log10(y[windows]), deg=1)[0]
            return p
        elif self.ndim == 2:
            N = np.shape(y)[0]
            p = np.zeros(N)
            for i in range(N):
                p[i] = np.polyfit(np.log10(w[windows]), np.log10(y[i][windows]), deg=1)[0]
            return p
        else:
            raise NotImplemented

    @property
    def Muv(self) -> float | np.ndarray | NotImplemented:
        '''
        Rest-frame UV absolute magnitude, computed in tophat window from 1450-1550 Angstroms.
        If the SED has no unit information, returns NotImplemented.
        '''
        if not self.units or self.redshift is None: 
            return NotImplemented
        w = self.wav_rest.to(Unit('angstrom')).value
        tophat = (w > 1450)&(w < 1550)

        if self._y_type != 'fnu':
            s = self.convert_units(yunit='Jy', inplace=False)
            y = s._y['total'].value
        else:
            y = self._y['total'].to(Unit('Jy')).value

        if self.ndim == 1:
            mUV = -2.5*np.log10((np.mean(y[tophat])/(1+self.redshift))/3631)
        elif self.ndim == 2:
            N = np.shape(y)[0]
            mUV = np.zeros(N)
            for i in range(N):
                mUV[i] = -2.5*np.log10((np.mean(y[i][tophat])/(1+self.redshift))/3631)
        return mUV - 5*(np.log10(self.luminosity_distance.to(Unit('pc')).value)-1)

    @property
    def properties(self) -> dict:
        '''Dictionary of derived SED properties'''
        return dict(beta=self.beta, Muv=self.Muv, Lbol=self.Lbol)

    # things to compute automatically:
    # wavelength, frequency, energy
    # Lnu, Llam, Fnu, Flam
    # bolometric luminosity
    # sed.measure_window_luminosity((1400.0 * Angstrom, 1600.0 * Angstrom))
    # sed.measure_balmer_break()
    # sed.measure_d4000()
    # sed.measure_beta(windows='Calzetti94')

    def plot(self, ax: mpl.axes.Axes = None, 
             x: str = 'wav_rest', 
             y: str = 'total', 
             yerr: str = None,
             step: bool = False, 
             xscale: str = None,
             yscale: str = None,
             xunit: str | Unit = None,
             yunit: str | Unit = None,
             xlim: tuple[float,float] = None, 
             ylim: tuple[float,float] = None, 
             verbose_labels: bool = False,
             show: bool = False, 
             save: bool = False, 
             eng: bool = False, 
             **kwargs) -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """

        Args:
            ax (mpl.axes.Axes)
                Matplotlib axes object to plot on. If None (default), a new figure is created.
            x (str)
                SED property to plot on the x-axis. Default: 'wav_rest'. 
                Accepts 'wav_rest', 'wav_obs', 'freq_rest', 'freq_obs'.
            y (str)
                SED property to plot on the y-axis. Default: 'fnu'.
                Accepts 'fnu', 'flam', 'Lnu', 'Llam'.
            step (bool)
                Whether to plot a step plot. Default: False.
            xscale (str)
                Scaling for the x-axis. Default: 'linear'.
            yscale (str)
                Scaling for the y-axis. Default: 'linear'.
            xunit (str or Unit)
                Unit for the x-axis. Default: None, i.e., interpreted automatically
                from the x-axis values. 
            yunit (str or Unit)
                Unit for the y-axis. Default: None, i.e., interpreted automatically
                from the y-axis values.
            xlim (tuple)
                Limits for the x-axis. Default: None, i.e., automatically determined.
            ylim (tuple)
                Limits for the y-axis. Default: None, i.e., automatically determined.
            verbose_labels (bool)
                Whether to use verbose axis labels. Default: False.
            show (bool)
                Whether to display the plot. Default: False.
            save (bool)
                Whether to save the plot. Default: False.
            **kwargs
                Additional keyword arguments passed to ``matplotlib.pyplot.plot``.
        """
        

        x_plot = getattr(self, x)
        y_plot = self[y]
        if yerr is not None:
            yerr_plot = self[yerr]

        if xlim is None:
            xmin, xmax = np.min(x_plot), np.max(x_plot)
        else:
            xmin, xmax = xlim   
        
        if ylim is None:
            ymin, ymax = np.min(y_plot), np.max(y_plot)
        else:
            ymin, ymax = ylim



        if eng: 
            x_plot = x_plot.to(u.m)
        else: 
            if xunit is None:
                if xmax.to(u.micron).value < 1:
                    xunit = u.angstrom
                else:#if xmax.to(u.micron).value < 100:
                    xunit = u.micron
                # else:
                #     xunit = u.mm
            elif isinstance(xunit, str):
                xunit = utils.unit_parser(xunit)
            elif isinstance(yunit, str):
                yunit = utils.unit_parser(yunit)

        x_plot = x_plot.to(xunit)
        # y_plot = y_plot.to(yunit)

        if self._y_type == 'fnu':
            ylabel = r'$f_{\nu}$'
        elif self._y_type == 'flam':
            ylabel = r'$f_{\lambda}$'
        else:
            raise Exception

        if x == 'wav_rest':
            if verbose_labels: xlabel = 'Rest Wavelength'
            else: xlabel = r'$\lambda_{\rm rest}$'
        elif x == 'wav_obs':
            if verbose_labels: xlabel = 'Observed Wavelength'
            else: xlabel = r'$\lambda_{\rm obs}$'
        elif x == 'freq_rest':
            if verbose_labels: xlabel = 'Rest Frequency'
            else: xlabel = r'$\nu_{\rm rest}$'
        elif x == 'freq_obs':
            if verbose_labels: xlabel = 'Observed Frequency'
            else: xlabel = r'$\nu_{\rm obs}$'
        
        yunitstr = y_plot.unit.to_string(format="latex_inline")
        if r'erg\,\mathring{A}^{-1}' in yunitstr:
            yunitstr = yunitstr.replace(r'\,\mathring{A}^{-1}', r'\,')
            yunitstr = yunitstr.replace(r'\,cm^{-2}', r'\,cm^{-2}\,\mathring{A}^{-1}')
        if r'\mathrm{\mu' in yunitstr:
            yunitstr = yunitstr.replace(r'\mathrm{\mu', r'\mu\mathrm{')
        ylabel +=  fr' [{yunitstr}]'
        
        xunitstr = x_plot.unit.to_string(format="latex_inline")
        if r'\mathrm{\mu' in xunitstr:
            xunitstr = xunitstr.replace(r'\mathrm{\mu', r'\mu\mathrm{')
        xlabel +=  fr' [{xunitstr}]'

    
        with plt.style.context('brisket.brisket'):
            if ax is None:
                fig, ax = plt.subplots(figsize=(5,2.5))
            else:
                fig = plt.gcf()
                fig.canvas.draw()

            if yerr is None:
                if step:
                    ax.step(x_plot, y_plot, where='mid', **kwargs)
                else:
                    ax.plot(x_plot, y_plot, **kwargs)
            else:
                if step:
                    ax.step(x_plot, y_plot, where='mid', **kwargs)
                    # ax.errorbar(...)
                else:
                    ax.errorbar(x_plot, y_plot, yerr=yerr_plot, **kwargs)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)



            if eng:
                ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m', places=1))
            
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if xscale is not None:
                ax.set_xscale(xscale)
            if yscale is not None:
                ax.set_yscale(yscale)

            if show:
                plt.show()

        return fig, ax


    def implements(np_function):
        # "Register an __array_function__ implementation for SED objects."
        def decorator(func):
            np_handled_array_functions[np_function] = func
            return func
        return decorator

    @implements(np.sum)
    def sum(sed: SED, axis: int | None = None) -> SED:
        "Implementation of np.sum for SED objects"
        newobj = deepcopy(sed)
        newobj._y = np.sum(sed._y, axis=axis)
        return newobj

    @implements(np.convolve)
    def convolve(sed: SED, kernel: np.ndarray, mode: str = 'full') -> SED:
        """
        Implementation of np.convolve for SED objects

        Args:
            sed (SED)
                SED object to convolve.
            kernel (np.ndarray)
                Convolution kernel.
            mode (str)
                Convolution mode, passed to np.convolve. Default: 'full'.

        """
        newobj = deepcopy(sed)
        newobj._y = np.convolve(sed._y, kernel, mode=mode)
        return newobj

    @implements(np.shape)
    def shape(sed: SED) -> tuple:
        """Implementation of np.shape for SED objects. 
        Returns the shape of the SED flux array.
        
        Args:
            sed (SED)
                SED object.
        """
        return np.shape(sed._y['total'])

    @property 
    def T(self):
        newobj = deepcopy(self)
        newobj._y = np.transpose(self._y)
        return newobj

    @property 
    def components(self):
        return list(self._y.keys())


    # def append(self):



if __name__ == '__main__':
    # Test SED object
    import astropy.units as u
    wav_rest = np.linspace(5e2, 1e4, 5000) * u.angstrom
    fnu = np.ones(len(wav_rest)) * u.uJy
    fnu[wav_rest<1216*u.angstrom] *= 0

    sed = SED(wav_rest=wav_rest, total=fnu, redshift=7, verbose=True)
    sed['young'] = fnu / 2

    # print(sed._x_implemented)
    # sed.convert_units(xunit=u.micron, yunit=u.mJy)
    # print(sed['wav_obs'])
    # print(sed['total'])

    # print(sed['young'])
    # print(sed.components)
    # print(sed._y['total'].unit)
    print(sed)



