import brisket
# from brisket import config
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    params = brisket.Params()
    params['redshift'] = 7 # brisket.FreeParam(7, 8)

    params.add_source('galaxy', model_func=brisket.models.CompositeStellarPopModel)
    params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
    params['galaxy']['logMstar'] = 10 # brisket.FreeParam(8, 10) # 10^{10} Msun
    params['galaxy']['zmet'] = 1 # solar metallicity
    params['galaxy'].add_sfh('constant', model_func=brisket.models.ConstantSFH)
    params['galaxy']['constant']['age_min'] = 0.001 # from 1 Myr
    params['galaxy']['constant']['age_max'] = 0.1 # to 100 Myr
    # params['galaxy']['constant']['logweight'] = 0 # low weight

    params.print_tree()

    obs = brisket.Observation('test1', verbose=True)
    obs.add_phot(filters=['f435w','f606w','f814w','f115w','f150w','f200w','f277w','f356w'])

    mod = brisket.Model(params, obs=obs, verbose=True)
        
