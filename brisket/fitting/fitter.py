'''
This module defines the Fitter class, which is the primary interface to fitting SEDs using brisket.
'''
from __future__ import annotations 

import numpy as np
import os
import time
import warnings
import sys
from copy import deepcopy

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # from ..utils.sed import SED
    from ..parameters import Params
    from .observation import Observation

from ..console import setup_logger
from .priors import PriorVolume
from ..models.core import Model


# detect if run through mpiexec/mpirun
try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    from mpi4py.futures import MPIPoolExecutor

except ImportError:
    rank = 0

class Fitter:
    def __init__(self, 
                 params: Params, 
                 obs: Observation, 
                 run: str, 
                 outdir: str = None,
                 n_posterior: int = 500, 
                 verbose: bool = False):

        if verbose:
            self.logger = setup_logger(__name__, 'INFO')
        else:
            self.logger = setup_logger(__name__, 'WARNING')

        self.verbose = verbose
        self._check_install()

        self.run = run
        self.obs = obs
        assert self.obs.ID is not None, "Observation must have associated ID to fit model"
        self.params = deepcopy(params)
        self.n_posterior = n_posterior


        # # Set up the directory structure for saving outputs.
        # if rank == 0:
        #     utils.make_dirs(run=run)

        # The base name for output files.
        
        self.outpath = f'brisket_results/{run}/'
        if outdir is not None:
            assert os.path.exists(outdir), "outdir doesn't exist"
            self.outpath = os.path.join(outdir, self.outpath) # append outdir to beginning of path
        
            
        # self.fname = f'brisket/results/{run}/{self.galaxy.ID}_'


        # A dictionary containing properties of the model to be saved.
        # self.results = {}

        # Set up the model which is to be fitted to the data.
        # self.fitted_model = FittedModel(galaxy, self.parameters,
        #                                time_calls=time_calls)

        # self._set_constants()
        self.ndim = self.params.ndim
        self.param_names = self.params.free_param_names
        # self.mirrors = self.parameters.free_param_mirrors   # Params which mirror a fitted param
        # self.transforms = self.parameters.free_param_transforms   # Params which are a transform of another param

        # self.prior = Prior(self.limits, self.pdfs, self.hypers)
        self.prior = PriorVolume(self.params.free_param_priors)

        # Initialize the ModelGalaxy object with a set of parameters randomly drawn from the prior cube
        self.params.update_from_vector(self.param_names, self.prior.sample())
        self.mod = Model(self.params, self.obs, verbose=verbose)

    def _update_model(self, x):
        self.params.update_from_vector(self.param_names, x)
        self.mod.update(self.params) 

    def _set_constants(self):
        """ Calculate constant factors used in the lnlike function. """

        if self.galaxy.photometry_exists:
            log_error_factors = np.log(2*np.pi*self.galaxy.photometry[:, 2]**2)
            self.K_phot = -0.5*np.sum(log_error_factors)
            self.inv_sigma_sq_phot = 1./self.galaxy.photometry[:, 2]**2

        # if self.galaxy.index_list is not None:
        #     log_error_factors = np.log(2*np.pi*self.galaxy.indices[:, 1]**2)
        #     self.K_ind = -0.5*np.sum(log_error_factors)
        #     self.inv_sigma_sq_ind = 1./self.galaxy.indices[:, 1]**2

    def lnlike(self, x, ndim=0, nparam=0):
        """ Returns the log-likelihood for a given parameter vector. """

        self._update_model(x)

        lnlike = 0.

        for i in range(self.obs.N_spec):
            spec_obs = self.obs.spec_list[i]
            spec_mod = self.mod.obs.spec_list[i]
            # TODO implement
            # lnlike += ...
        
        for i in range(self.obs.N_phot):
            ymod = self.mod.phot._y.value
            yobs = self.obs.phot._y.value
            yobs_err = self.obs.phot._yerr.value
            K_phot = -0.5*np.sum(np.log(2*np.pi*yobs_err**2))
            lnlike += K_phot - 0.5 * np.sum((yobs - ymod)**2 / (yobs_err**2))

        # Return zero likelihood if lnlike = nan (something went wrong).
        if np.isnan(lnlike):
            print("lnlike was nan, replaced with zero probability.")
            return -9.99*10**99

        return lnlike

    # def _lnlike_spec(self):
    #     """ Calculates the log-likelihood for spectroscopic data. This
    #     includes options for fitting flexible spectral calibration and
    #     covariant noise models. """

    #     model = self.model_galaxy.spectrum
    #     if any(np.isnan(model)):
    #         print(self.parameters.data)
    #         quit()

    #     # Calculate differences between model and observed spectrum
    #     diff = (self.galaxy.spectrum[:, 1] - model)
    #     # print(any(np.isnan(diff)))

    #     # if "noise" in list(self.fit_instructions):
    #     #     if self.galaxy.spec_cov is not None:
    #     #         raise ValueError("Noise modelling is not currently supported "
    #     #                          "with manually specified covariance matrix.")

    #     #     self.noise = noise_model(self.model_components["noise"],
    #     #                              self.galaxy, model)
    #     # else:
    #     self.noise = noise_model({}, self.galaxy, model)


    #     if self.noise.corellated:
    #         lnlike_spec = self.noise.gp.lnlikelihood(self.noise.diff)

    #         return lnlike_spec

    #     else:
    #         # Allow for calculation of chi-squared with direct input
    #         # covariance matrix - experimental!
    #         # if self.galaxy.spec_cov is not None:
    #         #     diff_cov = np.dot(diff.T, self.galaxy.spec_cov_inv)
    #         #     self.chisq_spec = np.dot(diff_cov, diff)

    #         #     return -0.5*self.chisq_spec

    #         self.chisq_spec = np.sum(self.noise.inv_var*diff**2)

    #         # if "noise" in list(self.fit_instructions):
    #         #     c_spec = -np.log(self.model_components["noise"]["scaling"])
    #         #     K_spec = self.galaxy.spectrum.shape[0]*c_spec

    #         # else:
    #         K_spec = 0.

    #         return K_spec - 0.5*self.chisq_spec

    # def _lnlike_indices(self):
    #     """ Calculates the log-likelihood for spectral indices. """

    #     diff = (self.galaxy.indices[:, 0] - self.model_galaxy.indices)**2
    #     self.chisq_ind = np.sum(diff*self.inv_sigma_sq_ind)

    #     return self.K_ind - 0.5*self.chisq_ind

    def fit(self, 
            sampler: str = "multinest", 
            verbose: bool = False, 
            overwrite: bool = False,
            **kwargs):
            # n_live=400, 
            # use_MPI=True, 
            # n_eff=0, discard_exploration=False,
            # n_networks=4, pool=1, nsteps=4, ):
        """ Fit the specified model to the input galaxy data.


        Args:
            sampler (str, optional)
                Which sampler to use. Options are "multinest", "nautilus", and "ultranest". 
                Defaults to "multinest".

            verbose (bool, optional)
                Whether to print progress updates. Default: False.
            
            **kwargs (optional)
                Additional keyword arguments to pass to the sampler.
                For multinest, these are:
                    n_live (int, optional)
                        Number of live points: reducing speeds up the code but may lead to unreliable results. Default: 400
                    use_MPI (bool, default: False)
                For nautilus, these are:
                    n_eff (int, optional)
                        Target minimum effective sample size. Default: 0
                    discard_exploration (bool, optional)
                        Whether to discard the exploration phase to get more accurate results. Default: False
                    n_networks (int, optinal)
                        Number of neural networks. Default: 4
                    pool (int, optional)
                        Pool size used for parallelization. Default: 1
                For ultranest, these are:
                    nsteps (int, default: 4)

        """
        
        if False:
            pass
        # If results already exist, just read them in and return the FitResults object. 
        # if os.path.exists(f'{self.fname}brisket_results.fits'):
            
        #     self.results = Results(file=filepath)
        #     return self.results


            # if rank == 0:
            #     self.logger.info(f'Fitting not performed as results have already been loaded from {self.fname}brisket_results.fits. To start over delete this file or change run.')

            # file = fits.open(f'{self.fname}brisket_results.fits')['RESULTS']
            # self.parameters.data = utils.str_to_dict(file.header['PARAMS'])
            # self.results['samples2d'] = file.data['samples2d']
            # self.results['lnlike'] = file.data['lnlike']
            # self.results['lnz'] = file.header['LNZ']
            # self.results['lnz_err'] = file.header['LNZ_ERR']
            # self.results["median"] = np.median(self.results['samples2d'], axis=0)
            # self.results["conf_int"] = np.percentile(self.results["samples2d"],
            #                                          (16, 84), axis=0)            
            # self._print_results()
            # self.posterior = Posterior(self.galaxy, run=self.run, n_samples=self.n_posterior, logger=self.logger)
            # return

        else:
            # Figure out which sampling algorithm to use
            sampler = sampler.lower()

            if (sampler == "multinest" and not self.pymultinest_available and
                    self.nautilus_available):
                sampler = "nautilus"
                self.logger.warning("MultiNest installation not available. Switching to UltraNest.")

            elif (sampler == "nautilus" and not self.nautilus_available and
                    self.pymultinest_available):
                sampler = "multinest"
                self.logger.warning("Nautilus installation not available. Switching to UltraNest.")

            elif sampler not in ["multinest", "nautilus", "ultranest"]:
                e = ValueError(f"Sampler {sampler} not supported.")
                self.logger.error(e)
                raise e

            elif not (self.pymultinest_available or self.nautilus_available or self.ultranest_available):
                e = RuntimeError("No sampling algorithm could be loaded.")
                self.logger.error(e)
                raise e

        if rank == 0 or not use_MPI:
            self.logger.info(f'Fitting object') # TODO implement object ID
            start_time = time.time()

        with warnings.catch_warnings():# and os.environ['PYTHONWARNINGS'] as 'ignore':
            warnings.simplefilter('ignore')

            if sampler == 'multinest':
                self._run_multinest(**kwargs)

            elif sampler == 'nautilus':
                self._run_nautilus(**kwargs)

            elif sampler == 'ultranest':
                self._run_ultranest(**kwargs)

        if rank == 0 or not use_MPI:
            runtime = time.time() - start_time
            if runtime > 60:
                runtime = f'{int(np.floor(runtime/60))}m{runtime-np.floor(runtime/60)*60:.1f}s'
            else: 
                runtime = f'{runtime:.1f} seconds'

            self.logger.info(f'Completed in {runtime}.')

            # Load sampler outputs 
            if sampler == "multinest":
                samples2d = np.loadtxt(self.fname + "post_equal_weights.dat")
                lnz_line = open(self.fname + "stats.dat").readline().split()
                self.results["samples2d"] = samples2d[:, :-1]
                self.results["lnlike"] = samples2d[:, -1]
                self.results["lnz"] = float(lnz_line[-3])
                self.results["lnz_err"] = float(lnz_line[-1])
                
                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == "nautilus":
                samples2d = np.zeros((0, self.fitted_model.ndim))
                log_l = np.zeros(0)
                while len(samples2d) < self.n_posterior:
                    result = n_sampler.posterior(equal_weight=True)
                    samples2d = np.vstack((samples2d, result[0]))
                    log_l = np.concatenate((log_l, result[2]))
                self.results["samples2d"] = samples2d
                self.results["lnlike"] = log_l
                self.results["lnz"] = n_sampler.log_z
                self.results["lnz_err"] = 1.0 / np.sqrt(n_sampler.n_eff)

                # clean up output from the sampler
                os.system(f'rm {self.fname}*')

            elif sampler == 'ultranest':
                self.results['samples2d'] = u_sampler.results['samples']
                self.results['lnlike'] = u_sampler.results['weighted_samples']['logl']
                self.results['lnz'] =  u_sampler.results['logz']
                self.results['lnz_err'] =  u_sampler.results['logzerr']
   
                # clean up output from the sampler
                os.system(f'rm -r ' + '/'.join(self.fname.split('/')[:-1]) + '/*')
            
            columns = []
            columns.append(fits.Column(name='samples2d', array=self.results['samples2d'], format=f'{self.fitted_model.ndim}D'))
            columns.append(fits.Column(name='lnlike', array=self.results['lnlike'], format='D'))
            hdu = fits.BinTableHDU.from_columns(fits.ColDefs(columns), 
                header=fits.Header({'EXTNAME':'RESULTS',
                                    'PARAMS':utils.dict_to_str(self.parameters.data),
                                    'LNZ':self.results['lnz'],
                                    'LNZ_ERR':self.results['lnz_err']}))
            hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
            hdulist.writeto(f'{self.fname}brisket_results.fits')

            self.results["median"] = np.median(self.results['samples2d'], axis=0)
            self.results["conf_int"] = np.percentile(self.results["samples2d"], (16, 84), axis=0)

            self._print_results()

            # Create a posterior object to hold the results of the fit.
            self.posterior = Posterior(self.galaxy, run=self.run,
                                       n_samples=self.n_posterior, logger=self.logger)

    def _run_multinest(self, n_live=400, use_MPI=False, **kwargs):
        self.pymultinest.run(
            self.lnlike,
            self.prior.transform,
            self.ndim, n_live_points=n_live,
            importance_nested_sampling=False, verbose=self.verbose,
            sampling_efficiency='model',
            outputfiles_basename=os.path.join(self.outpath, f'{self.obs.ID}_'), 
            use_MPI=use_MPI
        )

    def _run_nautilus(self):
        n_sampler = self.nautilus_sampler(
                self.fitted_model.prior.transform,
                self.fitted_model.lnlike, n_live=n_live,
                n_networks=n_networks, pool=pool,
                n_dim=self.fitted_model.ndim,
                filepath=self.fname + '.h5'
            )

        n_sampler.run(verbose=verbose, n_eff=n_eff,
                        discard_exploration=discard_exploration)

    def _run_ultranest(self):
         # os.environ['OMP_NUM_THREADS'] = '1'
        resume = 'resume'
        if overwrite:
            resume = 'overwrite'

        u_sampler = ReactiveNestedSampler(self.fitted_model.params, 
                                        self.fitted_model.lnlike, 
                                        transform=self.fitted_model.prior.transform, 
                                        log_dir='/'.join(self.fname.split('/')[:-1]), 
                                        resume=resume, 
                                        run_num=None)
        u_sampler.stepsampler = SliceSampler(nsteps=nsteps,#len(self.fitted_model.params)*4,
                                                generate_direction=generate_mixture_random_direction)
        u_sampler.run(
            min_num_live_points=n_live,
            dlogz=0.5, # desired accuracy on logz -- could allow to specify
            min_ess=self.n_posterior, # number of effective samples
            # update_interval_volume_fraction=0.4, # how often to update region
            # max_num_improvement_loops=3, # how many times to go back and improve
        )


    def _print_results(self):
        """ Print the 16th, 50th, 84th percentiles of the posterior. """

        parameter_len_max = np.max([len(p) for p in self.fitted_model.params])
        parameter_len = np.max([parameter_len_max+2, 25])

        if config.ascii:
            value_chars = ' ░▒▓█'
            border_chars = '═║╔╦╗╠╬╣╚╩╝'
        else:
            value_chars = ' ▁▂▃▄▅▆▇█'
            border_chars = '═║╔╦╗╠╬╣╚╩╝'

        self.logger.info(border_chars[2] + border_chars[0]*parameter_len + border_chars[3] + border_chars[0]*12 + border_chars[3] + border_chars[0]*12 + border_chars[3] + border_chars[0]*12 + border_chars[3] + border_chars[0]*54 + border_chars[4])
        self.logger.info(border_chars[1] + ' ' + 'Parameter' + ' '*(parameter_len-10) +border_chars[1]+'    16th    '+border_chars[1]+'    50th    '+border_chars[1]+'    84th    '+border_chars[1] + ' '*21 + 'Distribution' + ' '*21 + border_chars[1])
        self.logger.info(border_chars[5] + border_chars[0]*parameter_len + border_chars[6] + border_chars[0]*12 + border_chars[6] + border_chars[0]*12 + border_chars[6] + border_chars[0]*12 + border_chars[6] + border_chars[0]*54 + border_chars[7])
        for i in range(self.fitted_model.ndim):
            s = f"{border_chars[1]} "
            s += f"{self.fitted_model.params[i]}" + ' '*(parameter_len-len(self.fitted_model.params[i])-2) 
            s += f" {border_chars[1]} "
            
            p00 = self.fitted_model.prior.limits[i][0]
            p99 = self.fitted_model.prior.limits[i][1]
            p16 = self.results['conf_int'][0,i]
            p50 = self.results['median'][i]
            p84 = self.results['conf_int'][1,i]
            sig_digit = int(np.floor(np.log10(np.min([p84-p50,p50-p16]))))-1
            if sig_digit >= 0: 
                p00 = int(np.round(p00, -sig_digit))
                p16 = int(np.round(p16, -sig_digit))
                p50 = int(np.round(p50, -sig_digit))
                p84 = int(np.round(p84, -sig_digit))
                p99 = int(np.round(p99, -sig_digit))
                s += f"{p16:<10d} {border_chars[1]} {p50:<10d} {border_chars[1]} {p84:<10d}"
            else:
                p00 = np.round(p00, -sig_digit)
                p16 = np.round(p16, -sig_digit)
                p50 = np.round(p50, -sig_digit)
                p84 = np.round(p84, -sig_digit)
                p99 = np.round(p99, -sig_digit)
                s += f"{p16:<10} {border_chars[1]} {p50:<10} {border_chars[1]} {p84:<10}"

            s += f' {border_chars[1]} '

            bins = np.linspace(self.fitted_model.prior.limits[i][0], self.fitted_model.prior.limits[i][1], 39)
            ys, _ = np.histogram(self.results['samples2d'][i], bins=bins)
            ys = ys/np.max(ys)
            s += f"{p00:<7}"
            if config.ascii:
                for y in ys:
                    if y<1/8: s += value_chars[0]
                    elif y<3/8: s += value_chars[1]
                    elif y<5/8: s += value_chars[2]
                    elif y<7/8: s += value_chars[3]
                    else: s += value_chars[4]
            else:
                for y in ys:
                    if y<1/16: s += value_chars[0]
                    elif y<3/16: s += value_chars[1]
                    elif y<5/16: s += value_chars[2]
                    elif y<7/16: s += value_chars[3]
                    elif y<9/16: s += value_chars[4]
                    elif y<11/16: s += value_chars[5]
                    elif y<13/16: s += value_chars[6]
                    elif y<15/16: s += value_chars[7]
                    else: s += value_chars[8]
            s += f"{p99:>7}"
              
                    
            s += f' {border_chars[1]}'
        
            self.logger.info(s)
        self.logger.info(border_chars[8] + border_chars[0]*parameter_len + border_chars[9] + border_chars[0]*12 + border_chars[9] + border_chars[0]*12 + border_chars[9] + border_chars[0]*12 + border_chars[9] + border_chars[0]*54 + border_chars[10])


    def _check_install(self):
        from contextlib import redirect_stdout
        try:
            with open(os.devnull, "w") as f, redirect_stdout(f):
                import pymultinest
            self.pymultinest = pymultinest
            self.pymultinest_available = True
        except (ImportError, RuntimeError, SystemExit):
            self.logger.warning('PyMultiNest import failed, fitting will use the Ultranest sampler instead.')
            self.pymultinest_available = False

        try:
            from nautilus import Sampler
            self.nautilus_sampler = Sampler
            self.nautilus_available = True
        except (ImportError, RuntimeError, SystemExit):
            self.logger.warning('Nautilus import failed, fitting will use the Ultranest sampler instead.')
            self.nautilus_available = False

        try:
            from ultranest import ReactiveNestedSampler
            from ultranest.stepsampler import SliceSampler, generate_mixture_random_direction
            self.ultranest_available = True
        except (ImportError, RuntimeError, SystemExit):
            self.logger.warning('Ultranest import failed, fitting will use the Nautilus sampler instead.')
            self.ultranest_available = False

