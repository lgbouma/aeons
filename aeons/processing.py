"""
Processing:
    mask_quality_and_stitch
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

##############
# PROCESSING #
##############

def mask_quality_and_stitch(lcc):
    # mask quality flags and stitch
    time, flux, flux_err = [], [], []
    for lc in lcc:
        sel = (lc.quality == 0)
        time.append(lc[sel].time.value)
        flux_median = np.nanmedian(lc[sel].pdcsap_flux.value)
        flux.append(lc[sel].pdcsap_flux.value/flux_median)
        flux_err.append(lc[sel].pdcsap_flux_err.value/flux_median)

    time = np.hstack(time)
    flux = np.hstack(flux)
    flux_err = np.hstack(flux_err)

    return time, flux, flux_err
