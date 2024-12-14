from __future__ import annotations
import os, h5py
import numpy as np
from copy import deepcopy
from typing import Tuple

from .manager import GridManager
from .. import config
from ..utils.sed import SED

from scipy.interpolate import RegularGridInterpolator

class Grid:

    def __init__(self, name, bucket='brisket-data'):
        self.name = name
        if not name.endswith('.hdf5'):
            name += '.hdf5'

        gm = GridManager(bucket=bucket)
        gm.check_grid(name)
        self.path = os.path.join(config.grid_dir, name)
        
        self._load_from_hdf5(self.path)

    def _load_from_hdf5(self, path):
        with h5py.File(path, 'r') as f:
            axes = np.array(f['axes'][:],dtype=str)
            assert axes[-1] == 'wavs'
            self.wavs = f['wavs'][:]
            self.data = f['grid'][:]
            self.axes = axes[:-1]
            self.array_axes = axes
            for axis in self.axes:
                setattr(self, axis, f[axis][:])
    
    @property 
    def shape(self):
        return self.data.shape[:-1] # remove the last axis, which is the SED

    @property 
    def wavelengths(self):
        return self.wavs

    def __getitem__(self, indices):
        newgrid = deepcopy(self)
        newgrid.data = newgrid.data[indices]
        return newgrid
    
    def __setitem__(self, indices, values):
        self.data[indices] = values

    def __array__(self, dtype=None, copy=None):
        return self.data

    def __repr__(self):
        return f'Grid({self.name}, shape={self.shape})'


    def resample(self, new_wavs, fill=0):
        new_data = spectres.spectres(new_wavs, self.wavs, self.data, fill=fill, verbose=False) 
        self.wavs = new_wavs
        return self.new_data

    def get_nearest(self, x, return_nearest=False):
        '''
        Returns the value of the grid at the point nearest to the vector x.
        If return_nearest=True, also returns the vector of the nearest point.
        '''
        points = ()
        for axis in self.axes:
            points += (getattr(self, axis),)
        
        index = ()
        for i in range(len(x)):
            index += (np.argmin(np.abs(points[i]-x[i])),)

        if return_nearest:
            return self.data[index], [points[i][j] for i,j in enumerate(index)]
        else:
            return self.data[index]

    def interpolate(self, 
                    params: dict[str, float],
                    inplace: bool = False,
                    ) -> np.ndarray | Grid:
        '''
        Returns the value of the grid at an arbitrary vector x, via linear interpolation.
        '''

        # if the grid has been updated (i.e., collapsed), we need to reinitialize the interpolator
        if hasattr(self, '_interpolator'):
            if self._interpolator_ndim != len(self.axes):
                delattr(self, '_interpolator')

        if not hasattr(self, '_interpolator'):
            points = ()
            for axis in self.axes:
                points += (getattr(self, axis),)
            
            self._interpolator_axes = list(self.axes)
            self._interpolator_ndim = len(points)
            self._interpolator = RegularGridInterpolator(points, self.data, bounds_error=False, fill_value=None)


        x = [params.get(axis, None) for axis in self.axes]

        if inplace:
            for i in self._interpolator_axes:
                delattr(self, self.axes[i])
            self.axes = [a for i,a in enumerate(self.axes) if i not in self._interpolator_axes]
            self.array_axes = [a for i,a in enumerate(self.array_axes) if i not in self._interpolator_axes]

            self.data = self._interpolator(x)
            return 
        
        else:
            if len(x) == 1:
                return self._interpolator(x)[0]
            else:
                return self._interpolator(x)


    def collapse(self, 
                 axis: str | int | Tuple[str, ...] | Tuple[int, ...], 
                 weights: np.ndarray = None, 
                 inplace: bool = False):
        '''
        Collapses (i.e., sums) the grid, along a given axis (or axes). 
        Optionally, specify weights.

        Args:
            axis (str | int | Tuple[str, ...] | Tuple[int, ...]): The axis or axes to collapse over.
            weights (np.ndarray): Weights to apply to the grid before collapsing.
            inplace (bool): If True, the grid is updated in place. Otherwise, a new grid is returned.
        '''

        if isinstance(axis, str) or isinstance(axis, int):
            # collapse over a single axis
            axis = (axis,)

        if all(isinstance(a, str) for a in axis):
            axis_indices = [list(self.axes).index(a) for a in axis]
        elif all(isinstance(a, int) for a in axis):
            axis_indices = list(axes)
        else:
            raise ValueError('axis must be a string, an integer, or a tuple of strings or integers')


        collapse_ndim = len(axis_indices)

        if collapse_ndim == 1:
            if weights is None:
                weights = np.ones(self.shape[axis_indices[0]])
            assert weights.shape == self.shape[axis_indices[0]]

        else:
            if weights is None:
                weights = np.ones(self.shape)
            assert weights.shape == self.shape #tuple([self.shape[i] for i in axis_indices])

        weights = np.expand_dims(weights, axis=-1)



        if inplace:
            for i in axis_indices:
                delattr(self, self.axes[i])
            self.axes = [a for i,a in enumerate(self.axes) if i not in axis_indices]
            self.array_axes = [a for i,a in enumerate(self.array_axes) if i not in axis_indices]

            self.data = np.sum(self.data * weights, axis=tuple(axis_indices))
            return 
        
        else:
            return np.sum(self.data * weights, axis=tuple(axis_indices))
        

        # # np.sum(self.data[i, :index, :].T * weights_2d[i, :index].T, axis=1)

        #     if weights_2d[i, index-1:].sum() > 0.:
        #         weights_2d[:, index-1] *= old_weight
        #         old += np.sum(self.grid[i, index-1:, :].T * weights_2d[i, index-1:].T, axis=1)
        #         weights_2d[:, index-1] /= old_weight


    def to_sed(self, **kwargs):
        '''
        Converts the grid to a SED object.
        '''
        redshift = kwargs.get('redshift', None)
        verbose = kwargs.get('verbose', False)
        return SED(redshift=redshift, verbose=verbose, Llam=self.data*u.Lsun/u.angstrom, wav_rest=self.wavs*u.angstrom)


if __name__=="__main__":
    g0 = Grid('bc03_miles_chabrier_a50')
    g1 = Grid('bc03_miles_chabrier_a50_cloudy_cont')
    g2 = Grid('bc03_miles_chabrier_a50_cloudy_lines')

    weights = np.zeros(g0.shape)
    weights[2,5] = 1
    weights[2,4] = 1
    weights[2,3] = 2
    weights[2,2] = 3
    weights[2,1] = 3
    weights[2,0] = 3

    y0 = g0.collapse(('zmet','age'), weights=weights, inplace=False)

    weights = np.zeros(g1.shape)
    weights[2,5] = 1
    weights[2,4] = 1
    weights[2,3] = 2
    weights[2,2] = 3
    weights[2,1] = 3
    weights[2,0] = 3

    g1.collapse(('zmet','age'), weights=weights, inplace=True)
    y1 = g1.interpolate({'logU':-1}, inplace=False)
    g2.collapse(('zmet','age'), weights=weights, inplace=True)
    l2 = g2.interpolate({'logU':-1}, inplace=False)
    

    x = np.logspace(2, 4, 10000)
    y0 = np.interp(x, g0.wavs, y0)
    y1 = np.interp(x, g1.wavs, y1)
    y2 = np.zeros_like(x)
    from .cloudy import linelist
    sigma = 0.0003
    for i in range(len(linelist)):
        g = np.exp(-(x-linelist.wavs[i])**2/(2*(sigma*linelist.wavs[i])**2))
        integral = np.trapezoid(g, x) 
        if integral==0:
            continue
        g /= integral
        y2 += g*l2[i]
    y2 /= 1000

    import matplotlib.pyplot as plt
    plt.style.use('hba_default')

    fig, ax = plt.subplots()
    ax.plot(x, y0, label='stellar')
    ax.plot(x, y1, label='neb. cont')
    # ax.plot(x, y2, label='neb. lines')
    ax.plot(x, y0+y1+y2, label='total')

    # ax.loglog()
    plt.show()
    
    #g1 = Grid('bc03_miles_chabrier_a50_cloudy_cont')


    quit()


    from .cloudy import linelist
    
    g = Grid('bc03_miles_chabrier_a50_cloudy_lines')
    # i = np.where(np.isin(linelist.names, ['Hb','[OIII]5007','Ha','[NII]6583']))[0][0]
    # i = 
    L_Hb = g.data[:,:,:,np.where(np.isin(linelist.names, ['Hb']))[0][0]]
    L_OIII = g.data[:,:,:,np.where(np.isin(linelist.names, ['[OIII]5007']))[0][0]]
    L_Ha = g.data[:,:,:,np.where(np.isin(linelist.names, ['Ha']))[0][0]]
    L_NII = g.data[:,:,:,np.where(np.isin(linelist.names, ['[NII]6583']))[0][0]]

    L_Hb = L_Hb.flatten()
    L_OIII = L_OIII.flatten()
    L_Ha = L_Ha.flatten()
    L_NII = L_NII.flatten()

    fig, ax = plt.subplots()

    ax.scatter(L_NII/L_Ha, L_OIII/L_Hb, marker='.', lw=0, c='k', alpha=0.5)

    ax.loglog()
    ax.set_xlim(0.003, 10)
    ax.set_ylim(0.02, 25)
    plt.show()


    quit()

    g0 = Grid('bc03_miles_chabrier_a50')
    g1 = Grid('bc03_miles_chabrier_a50_cloudy_lines')
    g2 = Grid('bc03_miles_chabrier_a50_cloudy_cont')
    
    import matplotlib.pyplot as plt
    plt.style.use('hba_default')

    # nearest, _ = g1.get_nearest((0.05, 1e6, -0.8), return_nearest=True)
    interp0 = g0.interpolate((0.25, 8e6)) 
    interp1 = g1.interpolate((0.25, 8e6, -1.5))
    interp2 = g2.interpolate((0.25, 8e6, -1.5)) 

    x = np.logspace(2, 4, 10000)
    y0 = np.interp(x, g0.wavs, interp0)
    y2 = np.interp(x, g2.wavs, interp2)
    y1 = np.zeros_like(x)
    from .cloudy import linelist
    # cond = np.array([l.startswith('H  1') for l in linelist.cloudy_labels])

    sigma = 0.0003
    for i in range(len(linelist)):
        g = np.exp(-(x-linelist.wavs[i])**2/(2*(sigma*linelist.wavs[i])**2))
        integral = np.trapezoid(g, x) 
        if integral==0:
            continue
        g /= integral
        y1 += g*interp1[i]
    y1 /= 1000

    y = y0 + y1 + y2
    
    fig, ax = plt.subplots()

    ax.step(x, y0*x**2, where='mid', label='stellar')
    # ax.step(x, y1*x**2, where='mid', label='neb. lines')
    ax.step(x, y2*x**2, where='mid', label='neb. cont')
    ax.step(x, y*x**2, where='mid', label='total', color='k')
    # ax.scatter(linelist.wavs, nearest, label='nearest')
    # ax.scatter(linelist.wavs, interp, label='interp')
    ax.loglog()
    plt.show()



