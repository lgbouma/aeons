"""
For the stars that might be in Cep-Her, which of them have Kepler light curves
showing rotation signals that a young star should have?

Assess by downloading all available Kepler long-cadence data for them, and
measure the rotation periods.  (Also, run notch and cache its result, since
this will need to happen anyway when these stars are searched).
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

import os, pickle
import lightkurve as lk
import numpy as np, matplotlib.pyplot as plt, pandas as pd

from aeons.paths import DATADIR, PHOTDIR, LOCALDIR, RESULTSDIR
from aeons.getters import get_kepler_lightcurve
from aeons.processing import mask_quality_and_stitch

from cdips.lcproc.find_planets import (
    detrend_and_iterative_tls, plot_detrend_check,
    plot_iterative_planet_finding_results
)
from cdips.utils import lcutils as lcu
from cdips.lcproc import detrend as dtr

import multiprocessing as mp
nworkers = mp.cpu_count()

def main():

    csvpath = os.path.join(DATADIR, 'tables', 'tab_supp_CepHer_X_Kepler.csv')
    df = pd.read_csv(csvpath)
    kep_ids = np.array(df['kepid']).astype(str)
    fn = lambda x: 'KIC-'+x
    star_ids = [*map(fn, kep_ids)]

    outdir = os.path.join(RESULTSDIR, 'cepher_rotation')
    if not os.path.exists(outdir): os.mkdir(outdir)

    for star_id in np.sort(star_ids):

        LOGINFO(f"Starting {star_id}")

        lcdir = os.path.join(PHOTDIR, star_id)
        if not os.path.exists(lcdir): os.mkdir(lcdir)

        lcc = get_kepler_lightcurve(
            star_id.replace("-", " "), download_dir=lcdir
        )

        if type(lcc) != lk.lightcurve.KeplerLightCurve:
            lcc = [l for l in lcc if l.LABEL == star_id.replace("-", " ")]
        else:
            assert lcc.LABEL == star_id.replace("-", " ")
            lcc = [lcc]

        assert len(lcc) > 0
        time, flux, flux_err = mask_quality_and_stitch(lcc)
        a_90_10 = np.nanpercentile(flux, 90) - np.nanpercentile(flux, 10)

        dtr_dict = {
            'method':'best', 'break_tolerance':0.5, 'window_length':0.5
        }
        search_method = 'tls'
        cachepath = os.path.join(
            LOCALDIR, f"{star_id}_{search_method}_iterative_search.pkl"
        )

        if isinstance(cachepath, str):
            assert cachepath.endswith(".pkl")
            if os.path.exists(cachepath):
                LOGINFO(f"Found {cachepath}, loading results.")
                with open(cachepath, 'rb') as f:
                    d = pickle.load(f)
                return d

        dtrcachepath = cachepath.replace(".pkl", "_dtrcache.pkl")
        lsp_options = {'period_min':0.1, 'period_max':20}
        if os.path.exists(dtrcachepath):
            LOGINFO(f"Found {dtrcachepath}, loading results.")
            with open(dtrcachepath, 'rb') as f:
                d = pickle.load(f)
            search_time, search_flux, dtr_stages_dict = (
                d['search_time'], d['search_flux'], d['dtr_stages_dict']
            )
        else:
            # otherwise, run the detrending; cache results
            search_time, search_flux, dtr_stages_dict = (
                dtr.clean_rotationsignal_tess_singlesector_light_curve(
                    time, flux, magisflux=True, dtr_dict=dtr_dict,
                    lsp_dict=None, maskorbitedge=True, lsp_options=lsp_options,
                    verbose=True, slide_clip_lo=3, slide_clip_hi=5
                )
            )
            outdict = {
                'search_time':search_time,
                'search_flux':search_flux,
                'dtr_stages_dict':dtr_stages_dict,
                'a_90_10':a_90_10
            }
            with open(dtrcachepath, 'wb') as f:
                pickle.dump(outdict, f)
                LOGINFO(f"Wrote {dtrcachepath}")

        plot_detrend_check(
            star_id, outdir, dtr_dict,
            dtr_stages_dict, instrument='kepler'
        )

        LOGINFO(f"Finished {star_id}")

if __name__ == "__main__":
    main()
