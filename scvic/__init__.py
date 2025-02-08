__version__ = "1.0.1"

import logging
from logging import NullHandler

from scvic._settings import set_verbosity, set_seed

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())

set_verbosity(logging.INFO)

logger.propagate = False

__all__ = [
    "set_verbosity",
    "set_seed"
]