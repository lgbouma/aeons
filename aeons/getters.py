"""
Getters:
    get_kepler_lightcurve

Processing:
"""
#############
## LOGGING ##
#############

import logging
from astrobase import log_sub, log_fmt, log_date_fmt

DEBUG = False
if DEBUG:
    level = logging.DEBUG
else:
    level = logging.INFO
LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=level,
    style=log_sub,
    format=log_fmt,
    datefmt=log_date_fmt,
)

LOGDEBUG = LOGGER.debug
LOGINFO = LOGGER.info
LOGWARNING = LOGGER.warning
LOGERROR = LOGGER.error
LOGEXCEPTION = LOGGER.exception


#############
## IMPORTS ##
#############

import os
import numpy as np, pandas as pd, matplotlib.pyplot as plt

###########
# GETTERS #
###########

def get_kepler_lightcurve(star_id, download_dir=None):
    """
    args:
        star_id (str): identifier passed to lightkurve searcher.
        download_dir (str): directory for cacheing the light curves.

    returns:
        LightCurveCollection of Kepler light curves
    """

    import lightkurve as lk
    assert isinstance(download_dir, str)

    lcset = lk.search_lightcurve(star_id)

    sel = (lcset.author=='Kepler') & (lcset.exptime.value==1800)

    if np.any(sel):
        lcc = lcset[sel].download_all(download_dir=download_dir)

    else:
        LOGEXCEPTION(f'Failed to get Kepler light curves for {star_id}')
        return 0

    return lcc
