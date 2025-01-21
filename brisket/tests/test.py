import brisket
from brisket import config
config.R_default = 1500
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':


    params = brisket.Params()
    params['redshift'] = 7 # brisket.FreeParam(7, 8)

    params.add_source('galaxy', model_func=brisket.models.CompositeStellarPopModel)
    params['galaxy']['grids'] = 'bc03_miles_chabrier_a50' # a50 = 50 ages (native = 221 ages)
    params['galaxy']['logMstar'] = 10 # brisket.FreeParam(8, 10) # 10^{10} Msun
    params['galaxy']['zmet'] = 0.3 # solar metallicity
    params['galaxy'].add_sfh('constant', model_func=brisket.models.ConstantSFH)
    params['galaxy']['constant']['age_min'] = 0.005 # from 1 Myr
    params['galaxy']['constant']['age_max'] = 0.2 # to 100 Myr
    # params['galaxy']['constant']['logweight'] = 0 # low weight

    params['galaxy'].add_nebular(model_func=brisket.models.CloudyNebularModel)
    params['galaxy']['nebular']['line_grid'] = 'bc03_miles_chabrier_a50_cloudy_lines'
    params['galaxy']['nebular']['cont_grid'] = 'bc03_miles_chabrier_a50_cloudy_cont'
    params['galaxy']['nebular']['logU'] = -2.5



    params.add_igm()

    params.print_tree()

    # obs = brisket.Observation('test1', verbose=True)
    # obs.add_phot(filters=['f435w','f606w','f814w','f115w','f150w','f200w','f277w','f356w'])

    mod = brisket.Model(params, verbose=True)
    mod.sed.convert_units(xunit=u.micron, yunit=u.uJy, inplace=True)
    print(mod.sed)

    fig, ax = plt.subplots()
    ax.semilogx(mod.sed['wav_obs'], mod.sed['total'])
    ax.semilogx(mod.sed['wav_obs'], mod.sed['young'])
    ax.semilogx(mod.sed['wav_obs'], mod.sed['old'])
    ax.loglog(mod.sed['wav_obs'], mod.sed['nebular_cont'])
    ax.set_xlim(0.3, 20)
    # ax.set_xlim(2.5, 6)
    # ax.set_ylim(-1e-17, 1e-16)
    # # ax.set_ylim(1e-5, 1e5)
    # ax.set_ylim(28, 18)
    plt.show()






    # from brisket.grids.grids import Grid
    # from brisket.grids.cloudy import linelist

    # g = Grid('bc03_miles_chabrier_a50_cloudy_lines')
    # age_idx = np.where(g.age < 1e7)[0]
    # g.data = g.data[:,age_idx,:,:]
    # L_OII = g.data[:,:,:,np.where(np.isin(linelist.names, ['[OII]3726']))[0][0]] + g.data[:,:,:,np.where(np.isin(linelist.names, ['[OII]3729']))[0][0]]
    # L_Hb = g.data[:,:,:,np.where(np.isin(linelist.names, ['Hb']))[0][0]]
    # L_OIII = g.data[:,:,:,np.where(np.isin(linelist.names, ['[OIII]5007']))[0][0]]
    # L_Ha = g.data[:,:,:,np.where(np.isin(linelist.names, ['Ha']))[0][0]]
    # L_NII = g.data[:,:,:,np.where(np.isin(linelist.names, ['[NII]6583']))[0][0]]

    # L_OII = L_OII.flatten()
    # L_Hb = L_Hb.flatten()
    # L_OIII = L_OIII.flatten()
    # L_Ha = L_Ha.flatten()
    # L_NII = L_NII.flatten()

    # fig, ax = plt.subplots()

    # ax.scatter(L_NII/L_Ha*6583/6563, L_OIII/L_Hb*5007/4861, marker='o', lw=0, c='k', alpha=0.8)


    # from astropy.io import fits
    # bp_grid = '/Users/hba423/codes/bagpipes/bagpipes/models/grids/bpass_2.2.1_bin_imf135_300_nebular_line_grids.fits'

    # # Grid of line fluxes.
    # nhdus = len(fits.open(bp_grid))
    # pipes_line_grid = np.array([fits.open(bp_grid)[i].data[1:,1:] for i in range(1,nhdus)])

    # line_names = np.loadtxt('/Users/hba423/codes/bagpipes/bagpipes/models/grids/cloudy_lines.txt', dtype=str, delimiter=',')
    # i_OIII = np.where(line_names==['O  3  5006.84A'])[0][0]
    # L_OIII = np.take(pipes_line_grid, i_OIII, axis=-1).flatten()
    # i_OII = np.where(line_names==['Blnd  3729.00A'])[0][0]
    # L_OII = np.take(pipes_line_grid, i_OII, axis=-1).flatten()
    # i_Hb = np.where(line_names==['H  1  4861.33A'])[0][0]
    # L_Hb = np.take(pipes_line_grid, i_Hb, axis=-1).flatten()
    # i_Ha = np.where(line_names==['H  1  6562.81A'])[0][0]
    # L_Ha = np.take(pipes_line_grid, i_Ha, axis=-1).flatten()
    # i_NII = np.where(line_names==['N  2  6583.45A'])[0][0]
    # L_NII = np.take(pipes_line_grid, i_NII, axis=-1).flatten()

    # ax.scatter(L_NII/L_Ha, L_OIII/L_Hb, marker='o', lw=0, c='b', alpha=0.8)

    # files = ['nebular_emission_Z001.txt','nebular_emission_Z0001.txt','nebular_emission_Z002.txt','nebular_emission_Z0002.txt','nebular_emission_Z004.txt','nebular_emission_Z0005.txt','nebular_emission_Z006.txt','nebular_emission_Z008.txt','nebular_emission_Z010.txt','nebular_emission_Z014.txt','nebular_emission_Z017.txt','nebular_emission_Z020.txt','nebular_emission_Z030.txt','nebular_emission_Z040.txt']
    # for file in files:
    #     # log(Us)        xid       nh        (C/O)/(C/O)sol         mup      [OII]3727           Hb           [OIII]4959           [OIII]5007           [NII]6548           Ha           [NII]6584           [SII]6717           [SII]6731           NV1240           CIV1548           CIV1551           HeII1640           OIII]1661           OIII]1666           [SiIII]1883           SiIII]1888           CIII]1908 
    #     logU, xid, nh, CO, mup, L_OII, L_Hb, _, L_OIII, _, L_Ha, L_NII, _, _, _, _, _, _, _, _, _, _, _ = np.loadtxt(f'/Users/hba423/Library/CloudStorage/Dropbox/research/NIRSpec/models/nebular_emission_gutkin16/{file}', unpack=True)
    #     # cond = (nh == 100) & (xid == 0.3) & (CO == 1) & (mup == 100)
    #     cond = CO == 1
    #     ax.scatter(L_NII[cond]/L_Ha[cond], L_OIII[cond]/L_Hb[cond], marker='o', lw=0, c='r', alpha=0.5)


    # ax.loglog()
    # ax.set_xlim(0.003, 10)
    # ax.set_ylim(0.02, 25)
    # plt.show()

