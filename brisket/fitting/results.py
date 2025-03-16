'''
This module defines the Fitter class, which is the primary interface to fitting SEDs using brisket.
'''
from __future__ import annotations 

import numpy as np
import os, time, tqdm, sys, warnings
from copy import deepcopy

from rich.table import Table

from ..models.core import Model
from ..console import rich_str, PathHighlighter

# results = Results()
# results.map
# results.posterior
class Samples:
    def __init__(self, 
                 keys: list, 
                 samples2d: np.ndarray):

        self.keys = keys
        self.samples2d = samples2d
        for i in range(len(keys)):
            setattr(self, keys[i], samples2d[:,i])

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return np.shape(self.samples2d)[0]
    
    def __repr__(self):
        table = Table(title=f'Samples (N={len(self)})', title_justify='left')
        table.add_column("Parameter", justify="left", no_wrap=True)
        table.add_column("16th", style='bold #FFE4B5', justify='center', no_wrap=True)
        table.add_column("50th", style='bold #FFE4B5', justify='center', no_wrap=True)
        table.add_column("84th", style='bold #FFE4B5', justify='center', no_wrap=True)
        # table.add_column("Value", style='bold #FFE4B5', justify='center', no_wrap=True)
        h = PathHighlighter()

        for i in range(len(self.keys)):
            k = self.keys[i]
            v = getattr(self, k)
            # table.add_row(h(k), f'{type(v)}, {np.shape(v)}')#f'{v[0]:.3f}, {v[1]:.3f}, ..., {v[-2]:.3f}, {v[-1]:.3f}')
            try:
                qs = np.percentile(v, [16, 50, 84])
                table.add_row(h(k), f'{qs[0]:.3f}', f'{qs[1]:.3f}', f'{qs[2]:.3f}')
            except:
                table.add_row(h(k), f'{type(v)}, {np.shape(v)}', '', '')#f'{v[0]:.3f}, {v[1]:.3f}, ..., {v[-2]:.3f}, {v[-1]:.3f}')

        s = rich_str(table)
        return s



    @property
    def median(self):
        return np.median(self.samples2d, axis=0)
    
    @property 
    def cont_int(self):
        return np.percentile(self.samples2d, (16, 84), axis=0)

class MAP:
    def __init__(self, 
                 model: Model, 
                 param_names: list, 
                 sampler_output: dict):
        self.model = model
        self.sampler_output = sampler_output

        i = np.argmax(sampler_output['lnlike'])
        self.lnlike = sampler_output['lnlike'][i]

        self.model.params.update_from_vector(param_names, sampler_output['samples2d'][i])
        self.model.update(self.model.params) 
        self.sed = self.model.sed

        self.samples = Samples(param_names, sampler_output['samples2d'][i])

        # self.sed = deepcopy(self.model.sed)
        # self.sed.verbose = False
        # components = {key: np.zeros((n_samples, len(self.sed._x))) * self.sed._y_unit for key in self.sed.components}
        # for i in tqdm.tqdm(range(len(self.samples))):
        #     self.model.params.update_from_vector(param_names, self.samples2d[i])
        #     self.model.update(self.model.params) 
        #     for key in components:
        #         components[key][i] = self.model.sed[key]
        # for key in components:
        #     self.sed._y[key] = components[key]

        for key in self.sed.properties:
            self.samples[key] = self.sed.properties[key]

    def __repr__(self):
        return self.samples.__repr__()

class Posterior:
    def __init__(self, 
                 model: Model, 
                 param_names: list,
                 sampler_output: dict, 
                 n_samples: int):
        self.model = model
        self.sampler_output = sampler_output

        i = np.random.choice(sampler_output['samples2d'].shape[0], n_samples, replace=False)
        self.samples2d = sampler_output['samples2d'][i]
        self.lnlike = sampler_output['lnlike'][i]


        self.samples = Samples(param_names, self.samples2d)

        self.sed = deepcopy(self.model.sed)
        self.sed.verbose = False
        components = {key: np.zeros((n_samples, len(self.sed._x))) * self.sed._y_unit for key in self.sed.components}
        for i in tqdm.tqdm(range(len(self.samples))):
            self.model.params.update_from_vector(param_names, self.samples2d[i])
            self.model.update(self.model.params) 
            for key in components:
                components[key][i] = self.model.sed[key]
        for key in components:
            self.sed._y[key] = components[key]

        for key in self.sed.properties:
            self.samples[key] = self.sed.properties[key]
        # self.samples update w/ mod.sed.properties

    # self.samples = {}  # Store all posterior samples

    # # Add 1D posteriors for fitted params to the samples dictionary
    # for i in range(self.fitted_model.ndim):
    #     param_name = self.fitted_model.params[i]
    #     self.samples[param_name] = self.samples2d[self.indices, i]

        # self.fitted_model._update_model_galaxy(self.samples2d[0, :])
        # self.fitted_model.model_galaxy._compute_properties()
        # for key in self.fitted_model.model_galaxy.properties:
        #     try: # for arrays
        #         l = len(self.fitted_model.model_galaxy.properties[key])
        #         self.samples[key] = np.zeros((self.n_samples, l))
        #     except TypeError: # for keys with no len() (i.e., floats)
        #         self.samples[key] = np.zeros(self.n_samples)

    def __repr__(self):
        return self.samples.__repr__()

class Results:
    def __init__(self, 
                 model: Model,
                 param_names: list,
                 sampler_output: dict, 
                 run: str,
                 n_posterior: int = 1000,
                ):

        # if verbose:
        #     self.logger = setup_logger(__name__, 'INFO')
        # else:
        #     self.logger = setup_logger(__name__, 'WARNING')

        self.param_names = param_names
        self.sampler_output = sampler_output

        self.lnz = sampler_output['lnz']
        self.lnz_err = sampler_output['lnz_err']
        
        self.map = MAP(model, param_names, sampler_output)
        self.posterior = Posterior(model, param_names, sampler_output, n_samples=n_posterior)

        # self.samples2d = None
        # self.lnlike = None
        # self.lnz = None
        # self.lnz_err = None

    def __repr__(self):
        return self.map.__repr__() + '\n' + self.posterior.samples.__repr__()

    @classmethod
    def load(cls, filepath):
        pass
        # with fits.open(filepath) as hdul:
        #     results_hdu = hdul['RESULTS']
        #     self.samples2d = results_hdu.data['samples2d']
        #     self.lnlike = results_hdu.data['lnlike']
        #     self.lnz = results_hdu.header['LNZ']
        #     self.lnz_err = results_hdu.header['LNZ_ERR']
            
            
    def save_to_file(self, filepath):
        pass
        # columns = [
        #     fits.Column(name='samples2d', array=self.samples2d, format=f'{self.samples2d.shape[1]}D'),
        #     fits.Column(name='lnlike', array=self.lnlike, format='D')
        # ]
        # hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), 
        #     header=fits.Header({'EXTNAME':'RESULTS',
        #                         'LNZ':self.lnz,
        #                         'LNZ_ERR':self.lnz_err}))
        # hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        # hdulist.writeto(filepath, overwrite=True)



    # # If fewer than n_samples exist in posterior, reduce n_samples
    # if self.samples2d.shape[0] < self.n_samples:
    #     self.n_samples = self.samples2d.shape[0]

    # # Randomly choose points to generate posterior quantities
    # self.indices = np.random.choice(self.samples2d.shape[0],
    #                                 size=self.n_samples, replace=False)

    # self.samples = {}  # Store all posterior samples

    # # Add 1D posteriors for fitted params to the samples dictionary
    # for i in range(self.fitted_model.ndim):
    #     param_name = self.fitted_model.params[i]
    #     self.samples[param_name] = self.samples2d[self.indices, i]

    # self._compute_posterior_quantities()
        # self.fitted_model._update_model_galaxy(self.samples2d[0, :])
        # self.fitted_model.model_galaxy._compute_properties()
        # for key in self.fitted_model.model_galaxy.properties:
        #     try: # for arrays
        #         l = len(self.fitted_model.model_galaxy.properties[key])
        #         self.samples[key] = np.zeros((self.n_samples, l))
        #     except TypeError: # for keys with no len() (i.e., floats)
        #         self.samples[key] = np.zeros(self.n_samples)



