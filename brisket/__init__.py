'''
Core module for the brisket package.
'''

from . import example
from . import utils
from . import models
from . import fitting

from .fitting import priors
from .parameters import Params, FreeParam, FixedParam
from .models.core import Model
from .observation import Observation, Photometry, Spectrum
from .fitting.fitter import Fitter