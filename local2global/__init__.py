from pkg_resources import get_distribution, DistributionNotFound

from local2global.utils.utils import AlignmentProblem, WeightedAlignmentProblem, SVDAlignmentProblem
from local2global.utils import Patch

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = ''
