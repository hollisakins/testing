import brisket
from brisket import config
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # create a params object
    params = brisket.Params()
    params['redshift'] = 7 # brisket.FreeParam(7, 8)

    params.add_source('galaxy', model_func=brisket.models.CompositeStellarPopModel)
    params['galaxy']['logMstar'] = 10 # brisket.FreeParam(8, 11)
    params['galaxy']['zmet'] = 1 # solar metallicity
    params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
    params['galaxy'].add_sfh('constant', model_func=brisket.models.ConstantSFH)
    params['galaxy']['constant']['age_min'] = 0.1 # brisket.FreeParam(0.05, 0.15)
    params['galaxy']['constant']['age_max'] = 0.4 # brisket.FreeParam(0.4, 0.6)

    params.add_igm()
    # params['igm']['xhi'] = 0.9

    # params.add_calibration()
    # params['calibration']['R_curve'] = 'PRISM'
    # params['calibration']['oversample'] = 10
    # params['calibration']['f_LSF'] = 1.0
    params.print_tree()
    print(params.all_param_names)


    obs = brisket.Observation('test1', verbose=True)
    obs.add_phot(filters=['f606w','f814w','f115w','f150w','f200w','f277w','f356w','f444w'])#,
    #             fnu=[0, 0, 0.46, 0.63, 0.69, 1.26, 2.95, 3.13], 
     #            fnu_err=[0.01]*8)

    mod = brisket.Model(params, obs=obs, verbose=False)

    param_names = ['galaxy/logMstar','galaxy/constant/age_min','galaxy/constant/age_max']
    samples2d = np.array([np.random.normal(loc=10, scale=0.2, size=1000), np.random.normal(loc=0.1, scale=0.01, size=1000), np.random.normal(loc=0.4, scale=0.02, size=1000)]).T
    lnlike = np.ones(1000)

    res = brisket.fitting.results.Results(model=mod, param_names=param_names, sampler_output={'lnlike':lnlike, 'lnz':1, 'lnz_err':0.1, 'samples2d':samples2d}, 
                  run='test', n_posterior=1000)

    print(res)             
    quit()
    

    fitter = brisket.Fitter(params, obs, run='test', verbose=False)
    result = fitter.fit(sampler='ultranest')

    print(fitter.params.ndim)
    print(fitter.lnlike(x))


    # print(mod.sed.components)
    mod.sed.convert_units(yunit='ergscm2a')



    fig = brisket.plotting.Figure(title='Test', figsize=(6,4), nrows=1, ncols=1)
    ax = fig.add_subplot(xlabel='Wavelength', ylabel='Flux Density', 
                         xscale='log', yscale='log', 
                         xlim=(0.7, 10), ylim=(5e-22, 1e-18))
    mod.sed.plot(ax=ax, x='wav_obs', y='total', color='k')
    mod.sed.plot(ax=ax, x='wav_obs', y='old', color='r')
    mod.sed.plot(ax=ax, x='wav_obs', y='young', color='b')

    print(mod.phot['total'])
    fig.show()


    quit()


    # obs = brisket.Observation('test1', verbose=True)
    # obs.add_phot(filters=['f150w','f200w','f277w','f356w'], fnu=[1,1,1,1], fnu_err=[0.3]*4, )
    # quit()
    # obs.add_spec(wavs=np.linspace(0.5,5.3,500), wav_units='um')

    
    fig, ax = fitter.mod.sed.plot(x='wav_obs', y='fnu', 
                           xscale='log', yscale='log', 
                           xlim=(0.5, 10), xunit='micron', 
                           ylim=(1e-2, 1e2))
    fitter.mod.phot.plot(ax=ax, x='wav_obs', y='fnu', linewidth=0, marker='o', color='black')
    fitter.obs.phot.plot(ax=ax, x='wav_obs', y='fnu', yerr='fnu_err', linewidth=0, marker='o', color='r', elinewidth=1)
    # ax.scatter(mod.phot.wav.to(u.micron), mod.phot.flam)
    # ax.step(mod.spec.wavs.to(u.micron).value, mod.spec.flux.to(u.uJy), where='mid')
    plt.show()

    