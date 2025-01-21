'''
This module defines the ModelGalaxy class, which is the primary interface to construct model SEDs using brisket.
'''
from __future__ import annotations 

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..utils.sed import SED
    from ..parameters import Params

import numpy as np
import warnings
import os, time
from copy import deepcopy
import logging

import spectres
from astropy.constants import c as speed_of_light
import astropy.units as u

from .. import config
from ..utils.sed import SED
from ..console import setup_logger
from ..observation import Observation, Photometry, Spectrum


# from brisket import utils
# from brisket import filters
# from .. import plotting

# from brisket.models.stellar_model import StellarModel
# from brisket.models.nebular_model import NebularModel
# from brisket.models.star_formation_history import StarFormationHistoryModel
# from brisket.models.dust_emission_model import DustEmissionModel
# from brisket.models.dust_attenuation_model import DustAttenuationModel
# from brisket.models.accretion_disk_model import AccretionDiskModel
# from brisket.models.agn_line_model import AGNLineModel
# from brisket.models.igm_model import IGMModel
# from brisket.models.calibration import SpectralCalibrationModel
# from brisket.parameters import Params

class Model(object):
    """
    Model galaxy generation with BRISKET.

    Args:
        parameters (brisket.parameters.Params)
            Model parameters.
    
        filt_list (list, optional)
            A list of filter curves (default: None).
            Only needed if photometric output is desired (internal 
            model SED will be generated regardless).

        spec_wavs (list, optional)
            An array of spectroscopic wavelengths (default: None).
            Only needed if spectroscopic output is desired (internal 
            model SED will be generated regardless).

        verbose (bool, optional)
            Whether to print log messages (default: True).
    """
    def __init__(self, 
                 params: Params, 
                 obs: Observation = None,
                 verbose: bool = False):

        self.verbose = verbose
        if self.verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        # Handle the input parameters, whether provided in dictionary form or in Params object.
        # self.logger.debug(f'Parameters loaded')            
        self.params = deepcopy(params)
        self.obs = obs

        if self.params.ndim > 0:
            e = ValueError('Cannot create a model with free parameters. Please provide a set of fixed parameters.')
            self.logger.error(e)
            raise e

        assert 'redshift' in self.params, "Redshift must be specified for any model"
        self.redshift = float(self.params['redshift'])
        if self.redshift > config.max_redshift:
            raise ValueError("""Attempted to create a model with too high redshift. 
                            Please increase max_redshift in brisket/config.py 
                            before making this model.""")

        # # Create a spectral calibration object to handle spectroscopic calibration and resolution.
        # self.calib = False
        # if self.spec_output and 'calib' in self.components: 
        #     self.calib = SpectralCalibrationModel(self._spec_wavs, self.params, logger=logger)
        
        # Calculate optimal wavelength sampling for the model
        self.logger.info('Calculating optimal wavelength sampling for the model')
        self.wavelengths, self.R = self.get_wavelength_sampling()
        
        # Initialize the various models and resample to the internal, optimized wavelength grid
        self.logger.info('Initializing the model components')
        self.components = self.params.components
        for comp_name, comp_params in self.components.items(): 
            # initialize the model
            comp_params.model = comp_params.model_func(comp_params, verbose=self.verbose) 
            comp_params.model.resample(self.wavelengths) # resample the model

            subcomps = comp_params.components
            for subcomp_name, subcomp_params in subcomps.items():
                subcomp_params.model = subcomp_params.model_func(subcomp_params, parent=comp_params.model, verbose=self.verbose)
                subcomp_params.model.resample(self.wavelengths)
            
            # then validate that sub-components were added correctly (only used by certain models)
            comp_params.model.validate_components(comp_params)

        # self.sources = [n for n,t in self.component_types.items() if t == 'source']
        # self.reprocessors = [n for n,t in self.component_types.items() if t == 'reprocessor']
        # self.absorbers = [n for n,t in self.component_types.items() if t == 'absorber']
        

        self.logger.info('Computing the SED')
        # Compute the main SED 
        self.compute_sed() 

        # Compute observables
        self.compute_observables()

    def update(self, params):
        """ Update the model outputs (spectra, photometry) to reflect 
        new parameter values in the parameters dictionary. Note that 
        only the changing of numerical values is supported."""
        self.params.update(params)

        # Compute the internal full SED 
        self.compute_sed()

        # Compute observables
        self.compute_observables()


    def compute_sed(self):
        """ This method is the primary workhorse for ModelGalaxy. It combines the 
        models for the various emission and absorption processes to 
        generate the internal full galaxy SED held within the class. 
        The compute_photometry and compute_spectrum methods generate observables 
        using this internal full spectrum. """

        self.sed = SED(wav_rest=self.wavelengths, redshift=self.redshift, verbose=self.verbose)

        for comp_name, comp_params in self.components.items():
            model = comp_params.model
            if model.type == 'source':
                sed_incident = model.emit(comp_params)
                self.sed += sed_incident
                # sources can have sub-components, e.g. nebular emission or dust
                for subcomp_name, subcomp_params in comp_params.components.items():
                    submodel = subcomp_params.model
                    if submodel.type == 'reprocessor':
                        sed_transmitted, emission_params = submodel.absorb(self.sed, subcomp_params)
                        self.sed = sed_transmitted + submodel.emit(emission_params)

                    if submodel.type == 'absorber':
                        sed_transmitted = submodel.absorb(self.sed, subcomp_params)
                        self.sed = sed_transmitted
            
            # reprocessors and absorbers can be standalone models, as well
            if model.type == 'reprocessor':
                sed_transmitted, emission_params = model.absorb(self.sed, comp_params)
                self.sed = sed_transmitted + model.emit(emission_params)

            if model.type == 'absorber':
                sed_transmitted = model.absorb(self.sed, comp_params)
                self.sed = sed_transmitted

        self.sed.assign_units(xunit=u.angstrom, yunit=u.Lsun/u.angstrom, inplace=True)
        # # Optionally divide the model by a polynomial for calibration.
        # if "calib" in list(self.fit_instructions):
        #     self.calib = calib_model(self.model_components["calib"],
        #                              self.galaxy.spectrum,
        #                              self.model_galaxy.spectrum)

    def add_obs(self, obs: Observation):
        """ Add an observation to the model. This will resample the model to the 
        optimal wavelength grid based on the observation. """
        self.__init__(self.params, obs=self.obs, )
        pass

    def get_wavelength_sampling(self):
        """ Calculate the optimal wavelength sampling for the model
        given the required resolution values specified in the config
        file. The way this is done is key to the speed of the code. """
              
        max_wav = config.max_wavelength.to(u.angstrom).value
        wavelengths = [1.]
        R_default = config.R_default
        if self.obs is None:
            while wavelengths[-1] < max_wav:
                w = wavelengths[-1]
                wavelengths.append(w*(1.+0.5/R_default))
            wavelengths = np.array(wavelengths)
            R = np.zeros_like(wavelengths) + R_default

        else:
            sig = 3
            R_wav = np.logspace(0, np.log10(config.max_wavelength.to(u.angstrom).value), 1000)
            R = np.zeros_like(R_wav) + R_default

            for phot in self.obs.phot_list:
                wav_range = (phot.wav_range[0].to(u.angstrom).value, phot.wav_range[1].to(u.angstrom).value)
                in_phot_range = (R_wav > wav_range[0]/(1+config.max_redshift) * (1-25*sig/len(R_wav))) & (R_wav < wav_range[1]/(1+config.min_redshift) * (1+25*sig/len(R_wav)))
                R[(in_phot_range)&(R<phot.R*20)] = phot.R*20
            for spec in self.obs.spec_list:
                in_spec_range = (R_wav > spec.wav_range[0]/(1+config.max_redshift) * (1-25*sig/len(R_wav))) & (R_wav < spec.wav_range[1]/(1+config.min_redshift) * (1+25*sig/len(R_wav)))
                Rspec = np.min([spec.R*5, config.R_max])
                R[(in_spec_range)&(R<Rspec)] = Rspec

            from astropy.convolution import convolve
            w = np.arange(-5*sig, 5*sig+1)
            kernel = np.exp(-0.5*(w/sig)**2)
            R = convolve(R, kernel)
            R[R<R_default] = R_default
            
            while wavelengths[-1] < max_wav:
                w = wavelengths[-1]
                r = np.interp(w, R_wav, R)
                wavelengths.append(w*(1.+0.5/r))
                
            wavelengths = np.array(wavelengths)
            R = np.interp(wavelengths, R_wav, R)

            self.logger.info('Resampling the filter curves onto model wavelength grid')
            for phot in self.obs.phot_list:
                phot.filters.resample_filter_curves(wavelengths)

        return wavelengths, R

    def compute_observables(self):
        """ This method generates predictions for observed photometry.
        It resamples filter curves onto observed frame wavelengths and
        integrates over them to calculate photometric fluxes. """

        if self.obs is None:
            self.logger.info('No observation provided, skipping calculation of observables')
            return

        self.logger.info('Computing observables')
        self.prediction = Observation()
        for phot in self.obs.phot_list:
            if phot._y_key in ['fnu', 'flam']:
                k = phot._y_key
            else:
                self.logger.info('No reference observed photometry, providing prediction in fnu units')
                k = 'fnu'
            f = phot.filters.get_photometry(self.sed, which=k)
            args = {'filters': phot.filters, k: f, 'redshift': self.redshift}
            self.prediction.add_phot(**args)
            
        for spec in self.obs.spec_list:
            # .. apply spectral calibration here
            spectrum = deepcopy(self.sed)
            # for comp_name, comp_params in self.components.items():
            #     model = comp_params.model
            #     if model.type == 'calibration':
            #         spectrum = model.apply(comp_params, self.spec_wavs, spectrum)
            spectrum.resample(wav_obs=spec.wavs)
            
            args = {'wavs': spec.wavs}
            if spec.flux is not None: # if spec has flux, error:
                args[spec._y_key] = getattr(sed, spec._y_key) # then make sure to adopt the same flux specification
            else:
                args['fnu'] = spectrum.fnu
            self.prediction.add_spec(**args) 
        
    @property
    def phot(self):
        return self.prediction.phot

    @property
    def spec(self):
        return self.prediction.spec



    # def compute_spectrum(self, sed):
    #     """ This method generates predictions for observed spectroscopy.
    #     It optionally applies a Gaussian velocity dispersion then
    #     resamples onto the specified set of observed wavelengths. """

    #     # if "veldisp" in self.parameters:
    #     #     vres = 3*10**5/config.R_spec/2.
    #     #     sigma_pix = model_comp["veldisp"]/vres
    #     #     k_size = 4*int(sigma_pix+1)
    #     #     x_kernel_pix = np.arange(-k_size, k_size+1)

    #     #     kernel = np.exp(-(x_kernel_pix**2)/(2*sigma_pix**2))
    #     #     kernel /= np.trapz(kernel)  # Explicitly normalise kernel

    #     #     spectrum = np.convolve(self.spectrum_full, kernel, mode="valid")
    #     #     wav_obs = (1+self.redshift) * self.wavelengths[k_size:-k_size]

    #     # else:

    #     # if self.calib:
    #     #     if self.calib.R_curve is not None:
    #     #         self.logger.debug(f"Convolving output spectrum using provided R_curve")
    #     #         wav_obs, sed = self.calib.convolve_R_curve(wav_obs, sed, self.parameters['calib']['f_LSF'])

    #     spectrum = deepcopy(sed)

    #     for comp_name, comp_params in self.components.items():
    #         model = comp_params.model
    #         if model.type == 'calibration':
    #             spectrum = model.apply(comp_params, self.spec_wavs, spectrum)
        
    #     spectrum.resample(self._spec_wavs/(1+self.redshift))

    #     return spectrum

    # @property
    # def sed(self):
    #     if hasattr(self, '_sed_w_unitss'):
    #         return self._sed_w_units
    #     else:
    #         self._sed_w_units = self._sed.assign_units(xunit=u.angstrom, yunit=u.Lsun/u.angstrom, inplace=False)
    #     return sed

    # @property
    # def spectrum(self):
    #     return SED(wav_rest=self._spec_wavs/(1+self.redshift)*u.angstrom, Llam=self._spectrum._y, redshift=self.redshift, verbose=False)

    @property
    def photometry(self):
        return self._photometry

    def compute_properties(self): 

        self.properties = {}

        self.properties['redshift'] = self.redshift
        self.properties['t_hubble'] = config.age_at_z(self.redshift)

        # full SEDs (and spec, phot) for each component
        self.properties['SED'] = self.sed
        if self.phot_output: self.properties['photometry'] = self.compute_photometry(self.sed)
        if self.spec_output: self.properties['spectrum'] = self.compute_spectrum(self.sed)

        if len(self.components) >= 2:
            if 'galaxy' in self.components: 
                self.properties['SED_galaxy'] = self.sed_galaxy
                if self.phot_output: self.properties['phot_galaxy'] = self.compute_photometry(self.sed_galaxy)
                if self.spec_output: self.properties['spec_galaxy'] = self.compute_spectrum(self.sed_galaxy)
            if 'agn' in self.components: 
                self.properties['SED_accdisk'] = self.sed_accdisk
                self.properties['SED_AGN'] = self.sed_agn
                if self.phot_output: self.properties['phot_AGN'] = self.compute_photometry(self.sed_agn)
                if self.spec_output: self.properties['spec_AGN'] = self.compute_spectrum(self.sed_agn)
            if 'nebular' in self.components: 
                self.properties['SED_nebular'] = self.sed_nebular
                for line in self.nebular.line_grid:
                    self.properties[f'SED_nebular_{line}'] = self.nebular.line_grid[line]
                if self.phot_output: self.properties['phot_nebular'] = self.compute_photometry(self.sed_nebular)
                if self.spec_output: 
                    self.properties['spec_nebular'] = self.compute_spectrum(self.sed_nebular)
                    for line in self.nebular.line_grid:
                        self.properties[f'spec_nebular_{line}'] = self.compute_spectrum(self.nebular.line_grid[line])


        if 'galaxy' in self.components: 
            for q in ['stellar_mass', 'formed_mass', 'SFR_10', 'sSFR_10', 'nSFR_10', 'SFR_100', 'sSFR_100', 'nSFR_100', 'mass_weighted_age', 't_form', 't_quench']:
                self.properties[q] = getattr(self.galaxy.sfh, q)

            self.properties['SFH_ages'] = self.galaxy.sfh.ages
            self.properties['SFH_redshifts'] = config.z_at_age(self.properties['t_hubble']-self.properties['SFH_ages'])
            self.properties['SFH'] = self.galaxy.sfh.sfh


        # emission line fluxes, equivalent widths 
        # self.line_fluxes
        # self.line_EWs

        tophat = np.array((self.wavelengths > 1450)&(self.wavelengths < 1550),dtype=bool)
        if self.flam: 
            sed_flam = (self.sed*self.sed_units).to(u.erg/u.s/u.cm**2/u.angstrom).value
            sed_fnu = (self.sed*self.sed_units * (self.wav_obs * self.wav_units)**2 / speed_of_light).to(u.Jy).value
        else: 
            sed_flam = (self.sed*self.sed_units * speed_of_light / (self.wav_obs * self.wav_units)**2).to(u.erg/u.s/u.cm**2/u.angstrom).value
            sed_fnu = (self.sed*self.sed_units).to(u.Jy).value

        mUV = -2.5*np.log10(np.mean(sed_fnu[tophat])/(1+self.redshift)/3631)
        dL = config.cosmo.luminosity_distance(self.redshift).to(u.pc).value
        MUV = mUV - 5*(np.log10(dL)-1)
        self.properties['m_UV'] = round(mUV,4)
        self.properties['M_UV'] = round(MUV,4)
        
        # calzetti 1994 wavelength windows 
        windows = np.zeros(len(self.wavelengths),dtype=bool)
        windows |= (self.wavelengths>=1268)&(self.wavelengths<=1284)
        windows |= (self.wavelengths>=1309)&(self.wavelengths<=1316)
        windows |= (self.wavelengths>=1342)&(self.wavelengths<=1371)
        windows |= (self.wavelengths>=1407)&(self.wavelengths<=1515)
        windows |= (self.wavelengths>=1562)&(self.wavelengths<=1583)
        windows |= (self.wavelengths>=1677)&(self.wavelengths<=1740)
        windows |= (self.wavelengths>=1760)&(self.wavelengths<=1833)
        windows |= (self.wavelengths>=1866)&(self.wavelengths<=1890)
        windows |= (self.wavelengths>=1930)&(self.wavelengths<=1950)
        windows |= (self.wavelengths>=2400)&(self.wavelengths<=2580)
        p = np.polyfit(np.log10(self.wavelengths[windows]), np.log10(sed_flam[windows]), deg=1)
        self.properties['beta_UV'] = round(p[0],4)
        # beta_opt?

        # dust_curve
        # UVJ mags

        # # Set up a filter_set for calculating rest-frame UVJ magnitudes.
        # uvj_filt_list = np.loadtxt(utils.install_dir
        #                            + "/filters/UVJ.filt_list", dtype="str")
        # self.uvj_filter_set = filters.filter_set(uvj_filt_list)
        # self.uvj_filter_set.resample_filter_curves(self.wavelengths)


    def save_output(self, outfile, overwrite=True):
        assert hasattr(self, properties)

        if outfile.endswith('.fits'):
            from astropy.io import fits
            self.logger.info(f'Saving model output to {outfile}')
            tables = [fits.PrimaryHDU(header=fits.Header({'EXTEND':True}))]

            columns = []
            columns.append(fits.Column(name='wav_rest', array=self.wavelengths, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='wav_obs', array=self.wavelengths_obs, format='D', unit=str(self.wav_units)))
            columns.append(fits.Column(name='flux', array=self.spectrum_full, format='D', unit=str(self.sed_units)))
            tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'SED'})))

            if self.phot_output:
                columns = []
                columns.append(fits.Column(name='wav_obs', array=self.filter_set.eff_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_min', array=self.filter_set.min_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='wav_obs_max', array=self.filter_set.max_wavs, format='D', unit=str(self.wav_units)))
                columns.append(fits.Column(name='flux', array=self.photometry, format='D', unit=str(self.phot_units)))
                # tables.append(fits.BinTableHDU.from_columns(columns, header=fits.Header({'EXTNAME':'PHOT'})))

            t = fits.HDUList(tables)
            t.writeto(outfile, overwrite=overwrite)
