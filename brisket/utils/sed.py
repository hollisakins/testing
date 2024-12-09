"""
SED module for handling galaxy spectral energy distributions.


Example usage:

.. highlight:: python

::

    from brisket.utils.sed import SED
    sed = SED(wav_rest = wav, fnu = fnu, redshift=5)

"""
from __future__ import annotations
from collections.abc import Iterable

import sys
import numpy as np  
from copy import deepcopy
import astropy.units as u
from astropy.constants import c as speed_of_light
from astropy.constants import h as plancks_constant
import spectres

from .. import config
from . import utils
from .filters import Filters
border_chars = config.border_chars
from brisket.console import setup_logger

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
        units (bool, optional)
            Whether to use units (defaults to True). If True, the SED object
            expects wavelengths/fluxes to be provided with units via astropy.units, 
            and will assign the default units (specified in brisket.config) if none 
            are provided.
        **kwargs
            Flux specification: no more than one of 'Llam', 'Lnu', 'flam', 'fnu', 'nuLnu', 'lamLlam', 'nufnu', 'lamflam' 
            If none are provided, the SED will be populated with zeros (in L_lam).

    
    Attributes:
        wav_rest (array-like)
            The rest-frame wavelengths of the SED. 

    '''
    
    def __init__(self, 
                #  wav_rest: Iterable[float], 
                 redshift: float | None = None, 
                 verbose: bool = True, 
                 units: bool = True, 
                 **kwargs):
        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')


        self.units = units
        self.redshift = redshift
        if self.redshift is not None:
            if self.redshift == 0:
                self.luminosity_distance = 10 * u.pc
            else:
                self.luminosity_distance = config.cosmo.luminosity_distance(self.redshift).to(u.pc)
            if not units:
                self.luminosity_distance = self.luminosity_distance.value
        else:
            self.luminosity_distance = NotImplemented
        

        self._x_keys = ['wav_rest', 'wav_obs', 'freq_rest', 'freq_obs', 'energy_rest', 'energy_obs', 'filters']
        self._x_defs = [k in kwargs for k in self._x_keys]
        if sum(self._x_defs) != 1:
            self.logger.error(f"""Must supply exactly one specification of the SED 
                                  wavelength/frequency/energy axis. 
                                  Options are: {self._x_keys}""")
            sys.exit()

        self._x_key = self._x_keys[self._x_defs.index(True)]
        self._x = kwargs.get(self._x_key)

        if self._x_key == 'filters':

            if isinstance(self._x, Filters):
                self.filters = self._x
            elif isinstance(self._x, (list,np.ndarray)):
                self.filters = Filters(self._x)
            else:
                raise Exception
            self._x = self.filters.wav
            self._x_key = 'wav_obs'
        else:
            self.filters = None
            
        
        self._y_keys = ['Llam', 'Lnu', 'flam', 'fnu', 'nuLnu', 'lamLlam', 'nufnu', 'lamflam']
        self._y_defs = [k in kwargs for k in self._y_keys]
        if sum(self._y_defs) == 0:
            self.logger.info('No flux/luminosity information provided, populating with zeros. If this is intended, you can ignore this message.')
            self._y = np.zeros(len(self._x))
            self._y_key = self._y_keys[0]
        
        elif sum(self._y_defs) != 1:
            self.logger.error("Must supply at most one specification of the SED fluxes"); sys.exit()
        
        else:
            self._y_key = self._y_keys[self._y_defs.index(True)]
            self._y = kwargs.get(self._y_key)
            
            
        
        self._yerr_keys = ['err','error','Llam_err', 'Lnu_err', 'flam_err', 'fnu_err', 'nuLnu_err', 'lamLlam_err', 'nufnu_err', 'lamflam_err']
        self._yerr_defs = [k in kwargs for k in self._yerr_keys]
        if sum(self._yerr_defs) == 0:
            self.logger.info('No flux/luminosity error information provided, populating with zeros. If this is intended, you can ignore this message.')
            self._yerr = np.zeros(len(self._x))
            self._yerr_key = self._yerr_keys[0]

        elif sum(self._yerr_defs) != 1:
            self.logger.error("Must supply at most one specification of the SED fluxes"); sys.exit()
        
        else:
            self._yerr_key = self._yerr_keys[self._yerr_defs.index(True)]
            self._yerr = kwargs.get(self._yerr_key)
            

        if units:
            x_default_units = {'wav_rest':config.default_wavelength_unit, 
                               'wav_obs':config.default_wavelength_unit,
                               'freq_rest':config.default_frequency_unit,
                               'freq_obs':config.default_frequency_unit,
                               'energy_rest':config.default_energy_unit,
                               'energy_obs':config.default_energy_unit}
            if not hasattr(self._x, "unit"):
                self.logger.info(f"No units specified for {self._x_key}, adopting default ({x_default_units[self._x_key]})")
                self._x *= x_default_units[self._x_key]

            y_default_units = {'Llam': config.default_Llam_unit, 
                               'Lnu': config.default_Lnu_unit, 
                               'flam': config.default_flam_unit, 
                               'fnu': config.default_fnu_unit, 
                               'nuLnu': config.default_lum_unit, 
                               'lamLlam': config.default_lum_unit, 
                               'nufnu': config.default_flux_unit, 
                               'lamflam': config.default_flux_unit}

            if not hasattr(self._y, "unit"):
                self.logger.info(f"No units specified for {self._y_key}, adopting default ({y_default_units[self._y_key]})")
                self._y *= y_default_units[self._y_key]

            if not hasattr(self._yerr, "unit"):
                self.logger.info(f"No units specified for {self._yerr_key}, adopting default ({y_default_units[self._y_key]})")
                self._yerr *= y_default_units[self._y_key]

            # TODO handle case where units are provided for y but not yerr

            self._x_unit = self._x.unit
            self._y_unit = self._y.unit

    def __getitem__(self, indices):
        '''Allows access to the flux array via direct indexing of the SED object'''
        newobj = deepcopy(self)
        newobj._y = newobj._y[indices]
        return newobj
    
    def __setitem__(self, indices, values):
        '''Allows setting of the flux array via direct indexing of the SED object'''
        self._y[indices] = values

    @property 
    def wav_rest(self) -> u.Quantity | np.ndarray:
        '''Rest-frame wavelengths'''
        if self._x_key=='wav_rest': 
            return self._x
        elif self._x_key=='wav_obs': 
            if self.redshift is None:
                return NotImplemented
            return self._x / (1+self.redshift)
        elif self._x_key=='freq_rest': 
            if not self.units:
                return NotImplemented
            return (speed_of_light/self._x).to(config.default_wavelength_unit)
        elif self._x_key=='freq_obs': 
            if self.redshift is None or not self.units:
                return NotImplemented
            return (speed_of_light/(self._x/(1+self.redshift))).to(config.default_wavelength_unit)
        elif self._x_key=='energy_rest': 
            if not self.units:
                return NotImplemented
            return (plancks_constant * speed_of_light / self._x).to(config.default_wavelength_unit)
        elif self._x_key=='energy_obs': 
            if self.redshift is None or not self.units:
                return NotImplemented
            return (plancks_constant * speed_of_light / self._x / (1+self.redshift)).to(config.default_wavelength_unit)

    @property 
    def wav_obs(self) -> u.Quantity | np.ndarray:
        '''Observed-frame wavelengths'''
        if self._x_key=='wav_rest': 
            if self.redshift is None:
                return NotImplemented
            return self._x * (1+self.redshift)
        elif self._x_key=='wav_obs': 
            return self._x
        elif self._x_key=='freq_rest': 
            if self.redshift is None or not self.units:
                return NotImplemented
            return (speed_of_light/(self._x*(self.redshift))).to(config.default_wavelength_unit)
        elif self._x_key=='freq_obs': 
            if not self.units:
                return NotImplemented
            return (speed_of_light/self._x).to(config.default_wavelength_unit)
        elif self._x_key=='energy_rest': 
            if not self.units:
                return NotImplemented
            return NotImplemented
        elif self._x_key=='energy_obs': 
            if self.redshift is None or not self.units:
                return NotImplemented
            return NotImplemented

    @property 
    def freq_rest(self) -> u.Quantity:
        '''Rest-frame frequencies'''
        return (speed_of_light/self.wav_rest).to(config.default_frequency_unit)
    @property 
    def freq_obs(self) -> u.Quantity:
        '''Observed-frame frequencies'''
        return self.freq_rest / (1+self.redshift)
    @property 
    def energy_rest(self) -> u.Quantity:
        '''Rest-frame energies'''
        return (plancks_constant * self.freq_rest).to(config.default_energy_unit)
    @property 
    def energy_obs(self) -> u.Quantity:
        '''Observed-frame energies'''
        return self.energy_rest / (1+self.redshift)


    @property
    def fnu(self):
        '''
        Spectral flux density in terms of flux per unit frequency. Automatically converts from the flux specification defined at construction.
        '''
        if self._y_key=='fnu':
            return (self._y).to(config.default_fnu_unit)
        elif self._y_key=='flam':
            return (self._y * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._y_key=='Lnu':
            return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._y_key=='Llam':
            return (self._y / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._y_key=='L':
            return (self._y / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._y_key=='f':
            return (self._y / self.nu_obs).to(config.default_fnu_unit)
        else:
            raise Exception

    @property
    def fnu_err(self):
        '''
        Spectral flux density in terms of flux per unit frequency. Automatically converts from the flux specification defined at construction.
        '''
        if self._y_key=='fnu':
            return (self._yerr).to(config.default_fnu_unit)
        elif self._y_key=='flam':
            return (self._yerr * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._y_key=='Lnu':
            return (self._yerr / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._y_key=='Llam':
            return (self._yerr / (4*np.pi*self.luminosity_distance**2) * self.wav_obs**2 / speed_of_light).to(config.default_fnu_unit)
        elif self._y_key=='L':
            return (self._yerr / self.nu_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_fnu_unit)
        elif self._y_key=='f':
            return (self._yerr / self.nu_obs).to(config.default_fnu_unit)
        else:
            raise Exception
    
    @property
    def flam(self):
        '''
        Spectral flux density in terms of flux per unit wavelength. Automatically converts from the flux specification defined at construction.
        '''
        if self._y_key=='fnu':
            return (self._y / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._y_key=='flam':
            return (self._y).to(config.default_flam_unit)
        elif self._y_key=='Lnu':
            return (self._y / (4*np.pi*self.luminosity_distance**2) / self.wav_obs**2 * speed_of_light).to(config.default_flam_unit)
        elif self._y_key=='Llam':
            return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._y_key=='L':
            return (self._y / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._y_key=='f':
            return (self._y / self.lam_obs).to(config.default_flam_unit)
        else:
            raise Exception
    
    @property
    def Llam(self):
        '''
        Spectral flux density in terms of flux per unit wavelength. Automatically converts from the flux specification defined at construction.
        '''
        if self._y_key=='fnu':
            pass
        elif self._y_key=='flam':
            pass # return (self._y).to(config.default_flam_unit)
        elif self._y_key=='Lnu':
            return (self._y / self.wav_obs**2 * speed_of_light).to(config.default_Llam_unit)
        elif self._y_key=='Llam':
            return self._y
            # return (self._y / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._y_key=='L':
            pass # return (self._y / self.lam_obs / (4*np.pi*self.luminosity_distance**2)).to(config.default_flam_unit)
        elif self._y_key=='f':
            pass # return (self._y / self.lam_obs).to(config.default_flam_unit)
        else:
            raise Exception

    #TODO define flam, Lnu, etc

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
        return self._y

    def __repr__(self):
        if self.units:
            all_flux_defs = ['fnu','flam','Lnu','Llam','nufnu','lamflam','nuLnu','lamLlam']
            if self.filters is not None:
                wstr = f'filters: {", ".join(self.filters.nicknames)} {np.shape(self._x)}'
            else:
                wstr = f'{self._x_key}: [{self._x.value[0]:.2f}, {self._x.value[1]:.2f}, ..., {self._x.value[-2]:.2f}, {self._x.value[-1]:.2f}] {self._x_unit} {np.shape(self._x)}'
            if np.ndim(self._y) > 1:
                fstr1 = f'{self._y_key} (base): [...] {self._y_unit} {np.shape(self._y)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._y_key])) # TODO add check for flux_defs being implemented 
                betastr = f'beta: ?, Muv: ?, Lbol: ?'
            else:
                if len(self._y) > 4:
                    fstr1 = f'{self._y_key} (base): [{self._y.value[0]:.2f}, {self._y.value[1]:.2f}, ..., {self._y.value[-2]:.2f}, {self._y.value[-1]:.2f}] {self._y_unit} {np.shape(self._y)}'
                else:
                    fstr1 = f'{self._y_key} (base): [{self._y.value[0]:.2f}, {self._y.value[1]:.2f}, ..., {self._y.value[-2]:.2f}, {self._y.value[-1]:.2f}] {self._y_unit} {np.shape(self._y)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._y_key])) # TODO add check for flux_defs being implemented 
                betastr = f'beta: ?, Muv: ?, Lbol: ?'
                # betastr = f'beta: {self.beta:.2f}, Muv: {self.Muv:.1f}, Lbol: ?'
            width = config.cols-2
        else:
            all_flux_defs = ['fnu','flam','Lnu','Llam','nufnu','lamflam','nuLnu','lamLlam']
            w = self.wav_rest
            f = self._y
            wstr = f'wav_rest: [{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {np.shape(w)}'
            if np.ndim(f) > 1:
                fstr1 = f'{self._which_str} (base): [...] {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: ?, Muv: ?, Lbol: ?'
            else:
                fstr1 = f'{self._which_str} (base): [{f[0]:.2f}, {f[1]:.2f}, ..., {f[-2]:.2f}, {f[-1]:.2f}] {np.shape(f)}'
                fstr2 = '(available) ' + ', '.join(map(str,[a for a in all_flux_defs if a != self._which_str])) 
                betastr = f'beta: {self.beta:.2f}, Muv: {self.Muv:.1f}, Lbol: ?'
            width = config.cols-2
        # width = np.max([width, len(wstr)+4])
        # border_chars = '═║╔╦╗╠╬╣╚╩╝'
        outstr = border_chars[2] + border_chars[0]*width + border_chars[4]
        outstr += '\n' + border_chars[1] + 'BRISKET-SED'.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + wstr.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + fstr1.center(width) + border_chars[1]
        outstr += '\n' + border_chars[1] + fstr2.center(width) + border_chars[1]
        outstr += '\n' + border_chars[5] + border_chars[0]*width + border_chars[7]
        outstr += '\n' + border_chars[1] + betastr.center(width) + border_chars[1]
        outstr += '\n' + border_chars[8] + border_chars[0]*width + border_chars[10]
        return outstr
        # if np.ndim(f) == 1:
            # if len(self.wav_rest) > 4:
            #     wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
            #     fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnu.unit}'
            # l = max((len(wstr),len(fstr)))
            # wstr = wstr.ljust(l+3)
            # fstr = fstr.ljust(l+3)
            # wstr += str(np.shape(w))
            # fstr += str(np.shape(f))

            # return f'''BRISKET-SED: wav: {wstr}, flux {np.shape(f)}'''
        # elif np.ndim(f)==2:
        #     if len(self.wav_rest) > 4:
        #         wstr = f'[{w[0]:.2f}, {w[1]:.2f}, ..., {w[-2]:.2f}, {w[-1]:.2f}] {config.default_wavelength_unit}'
        #         fstr = f'[{f[0]:.1e}, {f[1]:.1e}, ..., {f[-2]:.1e}, {f[-1]:.1e}] {self.fnu.unit}'
        #     l = max((len(wstr),len(fstr)))
        #     wstr = wstr.ljust(l+3)
        #     fstr = fstr.ljust(l+3)
        #     wstr += str(np.shape(w))
        #     fstr += str(np.shape(f))

        #     return f'''BRISKET-SED: wav: {wstr}\n             fnu: {fstr}'''

    def __str__(self):
        return self.__repr__()

    def __add__(self, other: SED) -> SED:
        if not np.all(other._x==self._x):
            other.resample(**{self._x_key:self._x})
        
        if self.units:
            newobj = SED(**{self._x_key:self._x}, redshift=self.redshift, verbose=False)
            newobj._y_key = self._y_key
            newobj._y = self._y + getattr(other, self._y_key)
        else:
            newobj = SED(**{self._x_key:self._x}, redshift=self.redshift, verbose=False, units=False)
            newobj._y_key = self._y_key
            newobj._y = self._y + getattr(other, self._y_key)
            
        return newobj
    
    def __mul__(self, other: int | float | np.ndarray) -> SED: 
        newobj = deepcopy(self)
        newobj._y = self._y * other
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


    def to(self, unit, inplace=False):
        # if unit is wavelength or frquency, adjust x-units
        # if unit is flux or flux density, adjust y-units
        # if unit is tuple of (wavelength OR frequency, flux OR flux density), adjust both x and y-units
        pass

        
        # if 'spectral flux density' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Converting SED flux units to f_nu ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2 * (1 * self.wav_units)**2 / speed_of_light).to(self.sed_units).value
        # elif 'spectral flux density wav' in list(self.sed_units.physical_type):
        #     self.logger.debug(f"Keeping SED flux units in f_lam ({self.sed_units})")
        #     self.sed_unit_conv = (1*u.Lsun/u.angstrom/u.cm**2).to(self.sed_units).value

    def measure_window_luminosity(self, window):
        pass

    def measure_monochromatic_luminosity(self):
        pass
    def measure_slope(self):
        pass

    ########################################################################################################################
    @property
    def Lbol(self) -> u.Quantity:
        '''
        Bolometric luminosity of the SED.
        To be implemented.
        '''
        return None

    @property
    def beta(self) -> float | np.ndarray:
        '''
        UV spectral slope measured using the Calzetti et al. (1994) spectral windows.
        If the SED has no unit information, returns NotImplemented.
        '''
        if not self.units:
            return NotImplemented
        w = self.wav_rest.to(u.angstrom).value
        windows = ((w>=1268)&(w<=1284))|((w>=1309)&(w<=1316))|((w>=1342)&(w<=1371))|((w>=1407)&(w<=1515))|((w>=1562)&(w<=1583))|((w>=1677)&(w<=1740))|((w>=1760)&(w<=1833))|((w>=1866)&(w<=1890))|((w>=1930)&(w<=1950))|((w>=2400)&(w<=2580))
        p = np.polyfit(np.log10(w[windows]), np.log10(self.flam[windows].value), deg=1)
        return p[0]

    @property
    def Muv(self) -> float | np.ndarray | NotImplemented:
        '''
        Rest-frame UV absolute magnitude, computed in tophat window from 1450-1550 Angstroms.
        If the SED has no unit information, returns NotImplemented.
        '''
        if not self.units or self.redshift is None: 
            return NotImplemented
        w = self.wav_rest.to(u.angstrom).value
        tophat = (w > 1450)&(w < 1550)
        mUV = (np.mean(self.fnu[tophat])/(1+self.redshift)).to(u.ABmag).value
        return mUV - 5*(np.log10(self.luminosity_distance.to(u.pc).value)-1)

    @property
    def properties(self) -> dict:
        '''Dictionary of derived SED properties'''
        return dict(beta=self.beta, Lbol=self.Lbol)

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
             y: str = 'fnu', 
             yerr: str = None,
             step: bool = False, 
             xscale: str = None,
             yscale: str = None,
             xunit: str | u.Unit = None,
             yunit: str | u.Unit = None,
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
            xunit (str or u.Unit)
                Unit for the x-axis. Default: None, i.e., interpreted automatically
                from the x-axis values. 
            yunit (str or u.Unit)
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
        y_plot = getattr(self, y)    
        if yerr is not None:
            yerr_plot = getattr(self, yerr) 

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

        if y == 'fnu':
            ylabel = r'$f_{\nu}$'
        elif y == 'flam':
            ylabel = r'$f_{\lambda}$'

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
        return np.shape(sed._y)

    @property 
    def T(self):
        newobj = deepcopy(self)
        newobj._y = np.transpose(self._y)
        return newobj

        