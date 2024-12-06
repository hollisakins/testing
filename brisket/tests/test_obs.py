import brisket
from brisket import config
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

# create a params object
    params = brisket.Params()
    params['redshift'] = 7

    params.add_source('galaxy', model=brisket.models.GriddedStellarModel)
    params['galaxy']['logMstar'] = 10 # 10^{10} Msun
    params['galaxy']['zmet'] = 1 # solar metallicity
    params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
    params['galaxy'].add_sfh('constant', model=brisket.models.ConstantSFH)
    params['galaxy']['constant']['age_min'] = 0.01 # from 1 Myr
    params['galaxy']['constant']['age_max'] = 0.1 # to 10 Myr
    params['galaxy']['constant']['logweight'] = 0 # low weight

    params.add_igm()
    params['igm']['xhi'] = 0.9

    # params.add_calibration()
    # params['calibration']['R_curve'] = 'PRISM'
    # params['calibration']['oversample'] = 10
    # params['calibration']['f_LSF'] = 1.0

    params.print_tree()

    obs = brisket.Observation()
    obs.add_phot(filters=['f150w','f200w','f277w','f356w'])
    obs.add_spec(wavs=np.linspace(0.5,5.3,500), wav_units='um')

    mod = brisket.Model(params, obs, verbose=True)

    fig, ax = mod.sed.plot(x='wav_obs', xscale='log', yscale='log', xlim=(0.5, 10), xunit='micron', ylim=(1e-1, 1e2))
    ax.scatter(mod.phot.wav/1e4, mod.phot.fluxes)
    ax.step(mod.spec.wavs.to(u.micron).value, mod.spec.fluxes.fnu, where='mid')
    plt.show()