import matplotlib as mpl
import matplotlib.pyplot as plt
from . import config

# class Axes(mpl.axes.Axes):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

class Figure:

    def __init__(self, title=None, figsize=(4, 3), dpi=100, 
                 nrows=1, ncols=1, 
                 left=None, bottom=None, right=None, top=None, 
                 wspace=None, hspace=None, 
                 width_ratios=None, height_ratios=None, 
                 style='brisket',
                 **kwargs):

        self.style = style
        if self.style == 'brisket':
            self.style = config.install_dir + '/brisket.mplstyle'
        with plt.style.context(self.style):
            self.fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
            self.gs = mpl.gridspec.GridSpec(nrows, ncols, figure=self.fig,
                                    left=left, bottom=bottom, right=right, top=top, 
                                    wspace=wspace, hspace=hspace, 
                                    width_ratios=width_ratios, height_ratios=height_ratios)
            self.title = title
            self.fig.suptitle(title)
        # self.subplots = []

    def add_subplot(self, *indices, **kwargs):
        if indices == ():
            ax = self.fig.add_subplot(self.gs[0])
        else:
            ax = self.fig.add_subplot(self.gs[*indices])
        # self.subplots.append(ax)

        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        if 'xscale' in kwargs:
            ax.set_xscale(kwargs['xscale'])
        if 'yscale' in kwargs:
            ax.set_yscale(kwargs['yscale'])
        if 'xlim' in kwargs:
            ax.set_xlim(kwargs['xlim'])
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])

        return ax

    def plot(self, *args, **kwargs):
        pass

    def save(self, filename):
        self.savefig(filename)

    def show(self):
        # self.fig.show()
        plt.show()









def create_figure(title=None, 
                  label=None, 
                  xlabel=None, 
                  ylabel=None, 
                  xscale='linear', 
                  yscale='linear',
                  xlim=None, 
                  xunit=None, 
                  ylim=None, 
                  yunit=None, 
                  figsize=(4, 3), 
                  dpi=100):
    pass


    return fig, ax
