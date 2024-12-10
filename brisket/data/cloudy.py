import numpy as np
import os
from astropy.io import fits

if "CLOUDY_DATA_PATH" in list(os.environ):
    cloudy_data_path = os.environ["CLOUDY_DATA_PATH"]

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

except ImportError:
    rank = 0
    size = 1

age_lim = 3.*10**7


def mpi_split_array(array):
    """ Distributes array elements to cores when using mpi. """
    if size > 1: # If running on more than one core

        n_per_core = array.shape[0]//size

        # How many are left over after division between cores
        remainder = array.shape[0]%size

        if rank == 0:
            if remainder == 0:
                core_array = array[:n_per_core, ...]

            else:
                core_array = array[:n_per_core+1, ...]

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                comm.send(array[start:stop, ...], dest=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                comm.send(array[start:stop, ...], dest=i)

        if rank != 0:
            core_array = comm.recv(source=0)

    else:
        core_array = array

    return core_array


def mpi_combine_array(core_array, total_len):
    """ Combines array sections from different cores. """
    if size > 1: # If running on more than one core

        n_per_core = total_len//size

        # How many are left over after division between cores
        remainder = total_len%size

        if rank != 0:
            comm.send(core_array, dest=0)
            array = None

        if rank == 0:
            array = np.zeros([total_len] + list(core_array.shape[1:]))
            array[:core_array.shape[0], ...] = core_array

            for i in range(1, remainder):
                start = i*(n_per_core+1)
                stop = (i+1)*(n_per_core+1)
                array[start:stop, ...] = comm.recv(source=i)

            for i in range(np.max([1, remainder]), size):
                start = remainder+i*n_per_core
                stop = remainder+(i+1)*n_per_core
                array[start:stop, ...] = comm.recv(source=i)

        array = comm.bcast(array, root=0)

    else:
        array = core_array

    return array



def make_cloudy_input_file(age, zmet, logU, path):
    """ Makes an instructions file for cloudy. """

    # Copy file with emission line names to the correct directory
    if not os.path.exists(cloudy_data_path + "/pipes_cloudy_lines.txt"):
        os.system("cp " + utils.install_dir + "/models/grids/cloudy_lines.txt "
                  + cloudy_data_path + "/pipes_cloudy_lines.txt")

    logQ = logU + 51.576

    f = open(path + "/cloudy_temp_files/logU_" + "%.1f" % logU
             + "_zmet_" + "%.3f" % zmet + "/" + "%.5f" % age + ".in", "w+")

    # input ionizing spectrum
    f.write(f"table SED \"brisket_{filename}.sed\"\n")

    # parameters
    f.write("sphere\n")
    default_params = {
        "no_grain_scaling": False,
        # ionisation parameter
        "ionisation_parameter": None,
        # radius in log10 parsecs, only important for spherical geometry
        "radius": None,
        # covering factor. Keep as 1 as it is more efficient to simply combine
        # SEDs to get != 1.0 values
        "covering_factor": None,
        # K, if not provided the command is not used
        "stop_T": None,
        # if not provided the command is not used
        "stop_efrac": None,
        # log10(N_H/cm^2), if not provided the command is not used
        "stop_column_density": None,
        # K, if not provided the command is not used
        "T_floor": None,
        # Hydrogen density
        "hydrogen_density": None,
        # redshift, only necessary if CMB heating included
        "z": 0.0,
        # include CMB heating
        "CMB": None,
        # include cosmic rays
        "cosmic_rays": None,
        # include metals
        "metals": True,
        # include dust grains
        "grains": None,
        # the geometry
        "geometry": None,
        # constant density flag
        "constant_density": None,
        # constant pressure flag
        "constant_pressure": None,
        # relative resolution the saved continuum spectra
        "resolution": 1.0,
        # output abundances
        "output_abundances": None,
        # output continuum
        "output_cont": None,
        # output full list of all available lines
        "output_lines": None,
        # output linelist
        "output_linelist": None,
    }


    # add background continuum
    if params["cosmic_rays"] is not None:
        cinput.append("cosmic rays, background\n")

    if params["CMB"] is not None:
        cinput.append(f'CMB {params["z"]}\n')

    # define hydrogen density
    if params["hydrogen_density"] is not None:
        cinput.append(f"hden {np.log10(params['hydrogen_density'])} log\n")

    # constant density flag
    if params["constant_density"] is not None:
        cinput.append("constant density\n")

    # constant pressure flag
    if params["constant_pressure"] is not None:
        cinput.append("constant pressure\n")

    if (params["constant_density"] is not None) and (
        params["constant_pressure"] is not None
    ):
        raise InconsistentArguments(
            """Cannot specify both constant pressure and density"""
        )

    # covering factor
    if params["covering_factor"] is not None:
        cinput.append(f'covering factor {params["covering_factor"]} linear\n')

    # Processing commands
    if params["iterate_to_convergence"] is not None:
        cinput.append("iterate to convergence\n")

    if params["T_floor"] is not None:
        cinput.append(f'set temperature floor {params["T_floor"]} linear\n')

    if params["stop_T"] is not None:
        cinput.append(f'stop temperature {params["stop_T"]}K\n')

    if params["stop_efrac"] is not None:
        cinput.append(f'stop efrac {params["stop_efrac"]}\n')

    if params["stop_column_density"] is not None:
        cinput.append(f'stop column density {params["stop_column_density"]}\n')
        # For some horrible reason the above is ignored in favour of a
        # built in temperature stop (4000K) unless that is turned off.
        cinput.append("stop temperature off\n")


    f.write("cosmic rays background\n")
    f.write("hden 2.000 log\n")
    f.write("Q(H) = " + str("%.3f" % logQ) + " log\n")
    f.write("radius 19.000 log\n")
    f.write("abundances old solar 84\n")
    f.write("grains ISM\n")
    f.write("metals grains " + "%.3f" % zmet + "\n")

    # Nitrogen abundances, -0.22 is the depletion factor, the final term
    # is to offset the "metals grains" command.
    if np.log10(zmet) <= -0.63:
        nitrogen_abund = -4.57 + np.log10(zmet) - 0.22 - np.log10(zmet)

    else:
        nitrogen_abund = -3.94 + 2.*np.log10(zmet) - 0.22 - np.log10(zmet)

    elements = ["magnesium", "sulphur", "calcium", "carbon", "oxygen",
                "neon", "silicon", "argon", "iron", "nitrogen", "helium"]

    abundances = np.array([-5.12, -4.79, -8.16, -3.74, -3.29, -3.91, -5.45,
                           -5.44, -6.33, nitrogen_abund,
                           np.log10(0.08096 + 0.02618*zmet)])

    for i in range(len(elements)):
        f.write("element abundance " + elements[i] + " "
                + "%.2f" % abundances[i] + "\n")

    f.write("########################################\n")

    f.write("##### Stopping criteria #####\n")
    f.write("iterate to convergence\n")

    f.write("########################################\n")

    f.write("##### Output continuum and lines #####\n")
    f.write("set save prefix \"" + "%.5f" % age + "\"\n")
    f.write("save last outward continuum \".econ\" units microns\n")
    f.write("save last line list intrinsic absolute column"
            + " \".lines\" \"pipes_cloudy_lines.txt\"\n")

    f.write("########################################")

    f.close()


def run_cloudy_model(age, zmet, logU, path):
    """ Run an individual cloudy model. """

    make_cloudy_sed_file(age, zmet)
    make_cloudy_input_file(age, zmet, logU, path)
    os.chdir(path + "/cloudy_temp_files/"
             + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet)

    os.system(os.environ["CLOUDY_EXE"] + " -r " + "%.5f" % age)
    os.chdir("../../..")


def extract_cloudy_results(age, zmet, logU, path):
    """ Loads individual cloudy results from the output files and converts the
    units to L_sol/A for continuum, L_sol for lines. """

    cloudy_lines = np.loadtxt(path + "/cloudy_temp_files/"
                              + "logU_" + "%.1f" % logU
                              + "_zmet_" + "%.3f" % zmet + "/" + "%.5f" % age
                              + ".lines", usecols=(1),
                              delimiter="\t", skiprows=2)

    cloudy_cont = np.loadtxt(path + "/cloudy_temp_files/"
                             + "logU_" + "%.1f" % logU + "_zmet_"
                             + "%.3f" % zmet + "/" + "%.5f" % age + ".econ",
                             usecols=(0, 3, 8))[::-1, :]

    # wavelengths from microns to angstroms
    cloudy_cont[:, 0] *= 10**4

    # subtract lines from nebular continuum model
    cloudy_cont[:, 1] -= cloudy_cont[:, 2]

    # continuum from erg/s to erg/s/A.
    cloudy_cont[:, 1] /= cloudy_cont[:, 0]

    # Get bagpipes input spectrum: angstroms, erg/s/A
    input_spectrum = get_bagpipes_spectrum(age, zmet)

    # Total ionizing flux in the bagpipes model in erg/s
    ionizing_spec = input_spectrum[(input_spectrum[:, 0] <= 911.8), 1]
    ionizing_wavs = input_spectrum[(input_spectrum[:, 0] <= 911.8), 0]
    pipes_ionizing_flux = np.trapz(ionizing_spec, x=ionizing_wavs)

    # Total ionizing flux in the cloudy outputs in erg/s
    cloudy_ionizing_flux = np.sum(cloudy_lines) + np.trapz(cloudy_cont[:, 1],
                                                           x=cloudy_cont[:, 0])

    # Normalise cloudy fluxes to the level of the input bagpipes model
    cloudy_lines *= pipes_ionizing_flux/cloudy_ionizing_flux
    cloudy_cont[:, 1] *= pipes_ionizing_flux/cloudy_ionizing_flux

    # Convert cloudy fluxes from erg/s/A to L_sol/A
    cloudy_lines /= 3.826*10**33
    cloudy_cont[:, 1] /= 3.826*10**33

    nlines = config.wavelengths.shape[0]
    cloudy_cont_resampled = np.zeros((nlines, 2))

    # Resample the nebular continuum onto wavelengths of stellar models
    cloudy_cont_resampled[:, 0] = config.wavelengths
    cloudy_cont_resampled[:, 1] = np.interp(cloudy_cont_resampled[:, 0],
                                            cloudy_cont[:, 0],
                                            cloudy_cont[:, 1])

    return cloudy_cont_resampled[:, 1], cloudy_lines


def compile_cloudy_grid(path):

    line_wavs = np.loadtxt(utils.install_dir
                           + "/models/grids/cloudy_linewavs.txt")

    for logU in config.logU:
        for zmet in config.metallicities:

            print("logU: " + str(np.round(logU, 1))
                  + ", zmet: " + str(np.round(zmet, 4)))

            mask = (config.age_sampling < age_lim)
            contgrid = np.zeros((config.age_sampling[mask].shape[0]+1,
                                 config.wavelengths.shape[0]+1))

            contgrid[0, 1:] = config.wavelengths
            contgrid[1:, 0] = config.age_sampling[config.age_sampling < age_lim]

            linegrid = np.zeros((config.age_sampling[mask].shape[0]+1,
                                line_wavs.shape[0]+1))

            linegrid[0, 1:] = line_wavs
            linegrid[1:, 0] = config.age_sampling[mask]

            for i in range(config.age_sampling[mask].shape[0]):
                age = config.age_sampling[mask][i]
                cont_fluxes, line_fluxes = extract_cloudy_results(age*10**-9,
                                                                  zmet, logU,
                                                                  path)

                contgrid[i+1, 1:] = cont_fluxes
                linegrid[i+1, 1:] = line_fluxes

            if not os.path.exists(path + "/cloudy_temp_files/grids"):
                os.mkdir(path + "/cloudy_temp_files/grids")

            np.savetxt(path + "/cloudy_temp_files/grids/"
                       + "zmet_" + str(zmet) + "_logU_" + str(logU)
                       + ".neb_lines", linegrid)

            np.savetxt(path + "/cloudy_temp_files/grids/"
                       + "zmet_" + str(zmet) + "_logU_" + str(logU)
                       + ".neb_cont", contgrid)

    # Nebular grids
    list_of_hdus_lines = [fits.PrimaryHDU()]
    list_of_hdus_cont = [fits.PrimaryHDU()]

    for logU in config.logU:
        for zmet in config.metallicities:

            line_data = np.loadtxt(path + "/cloudy_temp_files/"
                                   + "grids/zmet_" + str(zmet)
                                   + "_logU_" + str(logU) + ".neb_lines")

            hdu_line = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
                                     + "%.1f" % logU, data=line_data)

            cont_data = np.loadtxt(path + "/cloudy_temp_files/"
                                   + "grids/zmet_" + str(zmet)
                                   + "_logU_" + str(logU) + ".neb_cont")

            hdu_cont = fits.ImageHDU(name="zmet_" + "%.3f" % zmet + "_logU_"
                                     + "%.1f" % logU, data=cont_data)

            list_of_hdus_lines.append(hdu_line)
            list_of_hdus_cont.append(hdu_cont)

    hdulist_lines = fits.HDUList(hdus=list_of_hdus_lines)
    hdulist_cont = fits.HDUList(hdus=list_of_hdus_cont)

    hdulist_lines.writeto(path + "/cloudy_temp_files"
                          + "/grids/bagpipes_nebular_line_grids.fits",
                          overwrite=True)

    hdulist_cont.writeto(path + "/cloudy_temp_files"
                         + "/grids/bagpipes_nebular_cont_grids.fits",
                         overwrite=True)


def run_cloudy_grid(path=None):
    """ Generate the whole grid of cloudy models and save to file. """

    if path is None:
        path = utils.working_dir

    if rank == 0 and not os.path.exists(path + "/cloudy_temp_files"):
        os.mkdir(path + "/cloudy_temp_files")

    ages = config.age_sampling[config.age_sampling < age_lim]

    n_models = config.logU.shape[0]*ages.shape[0]*config.metallicities.shape[0]

    params = np.zeros((n_models, 3))

    n = 0
    for i in range(config.logU.shape[0]):
        for j in range(config.metallicities.shape[0]):

            # Make directory to store cloudy inputs/outputs
            if rank == 0:
                if not os.path.exists(path + "/cloudy_temp_files/"
                                      + "logU_" + "%.1f" % config.logU[i]
                                      + "_zmet_" + "%.3f" % config.metallicities[j]):

                    os.mkdir(path + "/cloudy_temp_files/"
                             + "logU_" + "%.1f" % config.logU[i]
                             + "_zmet_" + "%.3f" % config.metallicities[j])

            # Populate array of parameter values
            for k in range(ages.shape[0]):

                params[n, 0] = ages[k]
                params[n, 1] = config.metallicities[j]
                params[n, 2] = config.logU[i]
                n += 1

    # Assign models to cores
    thread_nos = mpi_split_array(np.arange(n_models))

    # Run models assigned to this core
    for n in thread_nos:
        age = params[n, 0]
        zmet = params[n, 1]
        logU = params[n, 2]

        print("logU: " + str(np.round(logU, 1)) + ", zmet: "
              + str(np.round(zmet, 4)) + ", age: "
              + str(np.round(age*10**-9, 5)))

        run_cloudy_model(age*10**-9, zmet, logU, path)

    # Combine arrays of models assigned to cores, checks all is finished
    mpi_combine_array(thread_nos, n_models)

    # Put the final grid fits files together
    if rank == 0:
        compile_cloudy_grid(path)

def make_cloudy_sed_file(filename, wavs, fluxes):
    energy = 911.8/wavs
    fluxes *= 1.27688336e+15 # # Lsun/A to erg/s/Hz TODO is this correct?
    fluxes[fluxes <= 0] = np.power(10.,-99)
    energy = np.flip(energy)
    fluxes = np.flip(fluxes)
    np.savetxt(f"{cloudy_data_path}/SED/brisket_{filename}.sed", 
               np.array([energy, fluxes]).T, 
               header="Energy units: Rydbergs, Flux units: erg/s/Hz")

import argparse, h5py, tqdm
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        description='Run cloudy on a given input grid.')

    parser.add_argument('--grid', help='Input brisket-formatted grid file.', type=str)
    # parser.add_argument('--grid', help='Input brisket-formatted grid file.', type=str)
    #parser.add_argument('-p', '--param', help='Path to TOML parameter file', metavar='{param}.toml', type=str)
    #parser.add_argument('-r', '--run', help='Run name', metavar='{run}', type=str)
    #parser.add_argument('-o', '--outdir', metavar='{dir}/', help='Output directory (defaults to brisket/)', type=str) 
    #parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output (y/n)')  # on/off flag
    args = parser.parse_args()
    assert os.path.exists(args.grid), f"File {args.grid} not found."

    with h5py.File(args.grid, 'r') as f:
        axes = np.array(f['axes'][:],dtype=str)
        assert axes[-1] == 'wavs'
        wavs = f['wavs'][:]
        grid = f['grid'][:]
        axes = axes[:-1]
        shape = grid.shape[:-1] # remove the last axis, which is the SED


        for index in tqdm.tqdm(np.ndindex(shape)):
            # params = {str(axes[i]): float(f[axes[i]][:][index[i]]) for i in range(len(axes))}
            # filename = "_".join([f"{k[:3]}_{v:.3f}" for k,v in params.items()])
            filename = "_".join([k[:3]+str(i) for k,i in zip(axes, index)])
            make_cloudy_sed_file(filename, wavs, grid[index])

            # make_cloudy_input_file(age, zmet, logU, path)
            # os.chdir(path + "/cloudy_temp_files/"
            #         + "logU_" + "%.1f" % logU + "_zmet_" + "%.3f" % zmet)

            # os.system(os.environ["CLOUDY_EXE"] + " -r " + "%.5f" % age)
            # os.chdir("../../..")



            pass
            # print(index, grid[index])
        
        # for axis in axes:
        #     if axis != 'wavs':
        #         print(f[axis][:])




        # print(f['grid'].keys())

