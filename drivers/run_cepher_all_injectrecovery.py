"""
Grid over:
    _r_pl = 1 R_earth to 16 R_earth
    _pl_orbper = 2 to 64 days
how well can you recover the injected signals?

Usage:
    Update desired subset at USER: UPDATE HERE
    Then:
    $ python -u run_cepher_all_injectrecovery.py &> logs/ch_all_job3.log &
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

###########
# IMPORTS #
###########
import os, pickle
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy import units as u
from itertools import product
from copy import deepcopy

import lightkurve as lk

from aeons.paths import DATADIR, PHOTDIR, LOCALDIR, RESULTSDIR
from aeons.getters import get_kepler_lightcurve
from aeons.processing import mask_quality_and_stitch

from cdips.lcproc.find_planets import (
    detrend_and_iterative_tls, plot_detrend_check,
    plot_iterative_planet_finding_results
)
from cdips.utils import lcutils as lcu
from cdips.utils.mamajek import (
    get_interp_Rstar_from_BpmRp, get_interp_mass_from_rstar
)

import multiprocessing as mp
#nworkers = mp.cpu_count()
nworkers = 16

def main():

    # results from rotation period measurements, via
    # run_cepher_rotation_measurement -> plot_rotation_diagnostics
    csvpath = os.path.join(
        RESULTSDIR, 'cepher_rotation_diagnostics',
        'tab_supp_CepHer_X_Kepler_ls_periods.csv'
    )
    df = pd.read_csv(csvpath)
    df = df[df.selected]
    df = df.sort_values(by='bpmrp', ascending=False).reset_index(drop=True)

    # USER: UPDATE HERE!
    #df = df[:45] # job0
    #df = df[45:45+44] # job1
    #df = df[45+44:45+44+44] #job2
    df = df[45+44+44:] #job3

    # assumes ZAMS, so underestimates radii past BP-RP=1.2, ~=K2V.
    df['r_star'] = get_interp_Rstar_from_BpmRp(np.array(df.bpmrp))
    df['r_star_err'] = 0.1*df.r_star
    df.loc[df.bpmrp>1.2, 'r_star_err'] = 0.2*df.loc[df.bpmrp>1.2, 'r_star']

    mstars = [get_interp_mass_from_rstar(r_star) for r_star in
              np.array(df.r_star)]
    df['m_star'] = mstars
    df['m_star_err'] = 0.1*df.m_star

    np.random.seed(42)

    N_stars = len(df)
    N_injrecov_per_star = 10 # approx 2-week runtime

    ln_r_pl = np.random.uniform(low=np.log(0.5), high=np.log(16),
                                size=(N_injrecov_per_star, N_stars))
    ln_pl_orbper = np.random.uniform(low=np.log(2), high=np.log(64),
                                     size=(N_injrecov_per_star, N_stars))

    _r_pl = np.exp(ln_r_pl)
    _pl_orbper = np.exp(ln_pl_orbper)

    for ix, _star_id in zip(range(N_stars), df.star_id):

        for r_pl, pl_orbper in product(_r_pl[:,ix], _pl_orbper[:,ix]):

            star_id = _star_id + f"-synth-P{pl_orbper:.4f}d-Rp{r_pl:.4f}"
            search_method = 'tls'
            cachepath = os.path.join(
                LOCALDIR, f"{star_id}_{search_method}_iterative_search.pkl"
            )
            outpath = os.path.join(
                RESULTSDIR, "cepher_all_injectrecovery",
                f"{star_id}_{search_method}_recovery_result.csv"
            )

            if os.path.exists(outpath):
                LOGINFO(f"Found {outpath}, continue.")
                continue

            lcdir = os.path.join(PHOTDIR, _star_id)
            if not os.path.exists(lcdir): os.mkdir(lcdir)

            lcc = get_kepler_lightcurve(
                _star_id.replace("-", " "), download_dir=lcdir
            )

            if type(lcc) != lk.lightcurve.KeplerLightCurve:
                lcc = [l for l in lcc if l.LABEL == _star_id.replace("-", " ")]
            else:
                assert lcc.LABEL == _star_id.replace("-", " ")
                lcc = [lcc]
            assert len(lcc) > 0

            time, flux, flux_err = mask_quality_and_stitch(lcc)

            # injection-related steps
            r_star = float(df.loc[df.star_id == _star_id, 'r_star'])*u.Rsun
            rp_rs = (r_pl*u.R_earth / r_star).cgs.value
            epoch = np.nanmin(time)+np.random.uniform(low=0,high=pl_orbper)
            depth = rp_rs**2
            inj_dict = {
                'star_id': _star_id,
                'period':pl_orbper,
                'epoch':epoch,
                'depth':depth,
                'r_pl':r_pl, # R_earth
                'r_star':r_star.to(u.Rsun).value,
                'rp_rs':rp_rs
            }
            time, flux, _ = lcu.inject_transit_signal(time, flux, inj_dict)

            dtr_dict = {
                'method':'best', 'break_tolerance':0.5, 'window_length':0.5
            }

            bpmrp = float(df.loc[df.star_id == _star_id, 'bpmrp'])

            R_star = float(df.loc[df.star_id == _star_id, 'r_star'])
            R_star_err = float(df.loc[df.star_id == _star_id, 'r_star_err'])
            RSTARMIN = 0.5 if bpmrp < 2.3 else 0.2
            R_star_min = max([R_star - 4*R_star_err, RSTARMIN])
            R_star_max = min([R_star + 4*R_star_err, 2.0])

            MSTARMIN = 0.5 if bpmrp < 2.1 else 0.2
            M_star = float(df.loc[df.star_id == _star_id, 'm_star'])
            M_star_err = float(df.loc[df.star_id == _star_id, 'm_star_err'])
            M_star_min = max([M_star - 4*M_star_err, MSTARMIN])
            M_star_max = min([M_star + 4*M_star_err, 2.0])

            outdicts = (
                detrend_and_iterative_tls(
                    star_id, time, flux, dtr_dict,
                    period_min=1, period_max=100,
                    R_star=R_star, M_star=M_star,
                    R_star_min=R_star_min, R_star_max=R_star_max,
                    M_star_min=M_star_min, M_star_max=M_star_max,
                    n_transits_min=5,
                    search_method=search_method, n_threads=nworkers,
                    magisflux=True, cachepath=cachepath,
                    slide_clip_lo=6, slide_clip_hi=3,
                    verbose=True
                )
            )

            outdir = os.path.join(RESULTSDIR, 'cepher_all_injectrecovery')
            if not os.path.exists(outdir): os.mkdir(outdir)

            recov, weakrecov = lcu.determine_if_recovered(cachepath, inj_dict)
            inj_dict['recovered'] = recov
            inj_dict['partial_recovered'] = weakrecov

            outdf = pd.DataFrame(inj_dict, index=[0])
            outdf.to_csv(outpath, index=False)
            LOGINFO(f'made {outpath}')

            dtr_stages_dict = outdicts['dtr_stages_dict']

            MAKE_DETREND_CHECK = 0
            if MAKE_DETREND_CHECK:
                plot_detrend_check(
                    star_id, outdir, dtr_dict,
                    dtr_stages_dict, r=outdicts[0]['r'], instrument='kepler'
                )

            OVERWRITE = 0
            plot_iterative_planet_finding_results(
                star_id, search_method, outdir, cachepath, dtr_dict,
                overwrite=OVERWRITE
            )


if __name__ == "__main__":
    main()
