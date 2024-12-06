'''
Core module for the brisket package.
'''

from . import example
from . import utils
from . import models
from . import fitting

from .parameters import Params
from .models.core import Model
from .fitting.observation import Observation