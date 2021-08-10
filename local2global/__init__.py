from pkg_resources import get_distribution, DistributionNotFound

from .utils import Patch, AlignmentProblem, WeightedAlignmentProblem


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = ''
