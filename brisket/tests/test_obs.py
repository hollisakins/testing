import brisket
from brisket import config
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

# create a params object
    params = brisket.Params()
    params['redshift'] = 7 # brisket.FreeParam(7, 8)

    params.add_source('galaxy', model=brisket.models.GriddedStellarModel)
    params['galaxy']['logMstar'] = 10 # brisket.FreeParam(8, 10) # 10^{10} Msun
    params['galaxy']['zmet'] = 1 # solar metallicity
    params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
    params['galaxy'].add_sfh('constant', model=brisket.models.ConstantSFH)
    params['galaxy']['constant']['age_min'] = 0.001 # from 1 Myr
    params['galaxy']['constant']['age_max'] = 0.1 # to 10 Myr
    params['galaxy']['constant']['logweight'] = 0 # low weight

    # params.add_igm()
    # params['igm']['xhi'] = 0.9

    # params.add_calibration()
    # params['calibration']['R_curve'] = 'PRISM'
    # params['calibration']['oversample'] = 10
    # params['calibration']['f_LSF'] = 1.0
    params.print_tree()

    obs = brisket.Observation('test1', verbose=True)
    obs.add_phot(filters=['f435w','f606w','f814w','f115w','f150w','f200w','f277w','f356w'])

    mod = brisket.Model(params, obs=obs, verbose=True)

    fig, ax = mod.sed.plot(x='wav_obs', y='fnu', 
                           xscale='log', yscale='log', 
                           xlim=(0.5, 10), xunit='micron', 
                           ylim=(1e-2, 1e2))
    # ax.scatter(mod.phot.wav.to(u.micron), mod.phot.flam)
    # ax.step(mod.spec.wavs.to(u.micron).value, mod.spec.flux.to(u.uJy), where='mid')
    plt.show()


    quit()



    # obs = brisket.Observation('test1', verbose=True)
    # obs.add_phot(filters=['f150w','f200w','f277w','f356w'], fnu=[1,1,1,1], fnu_err=[0.3]*4, )
    # quit()
    # obs.add_spec(wavs=np.linspace(0.5,5.3,500), wav_units='um')

    # fitter = brisket.Fitter(params, obs, run='test', verbose=False)
    result = fitter.fit(sampler='multinest', n_live=50)
    quit()

    x = fitter.prior.sample()
    print(x)
    print(fitter.params.ndim)
    print(fitter.lnlike(x))

    
    fig, ax = fitter.mod.sed.plot(x='wav_obs', y='fnu', 
                           xscale='log', yscale='log', 
                           xlim=(0.5, 10), xunit='micron', 
                           ylim=(1e-2, 1e2))
    fitter.mod.phot.plot(ax=ax, x='wav_obs', y='fnu', linewidth=0, marker='o', color='black')
    fitter.obs.phot.plot(ax=ax, x='wav_obs', y='fnu', yerr='fnu_err', linewidth=0, marker='o', color='r', elinewidth=1)
    # ax.scatter(mod.phot.wav.to(u.micron), mod.phot.flam)
    # ax.step(mod.spec.wavs.to(u.micron).value, mod.spec.flux.to(u.uJy), where='mid')
    plt.show()

    