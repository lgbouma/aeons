"""

"""
import os, pickle
import numpy as np, matplotlib.pyplot as plt, pandas as pd
from astropy import units as u
from itertools import product
from copy import deepcopy

from aeons.paths import DATADIR, PHOTDIR, LOCALDIR, RESULTSDIR
from aeons.getters import get_kepler_lightcurve
from aeons.processing import mask_quality_and_stitch

from cdips.lcproc.find_planets import (
    detrend_and_iterative_tls, plot_detrend_check,
    plot_iterative_planet_finding_results
)
from cdips.utils import lcutils as lcu

import multiprocessing as mp
nworkers = mp.cpu_count()

def main():

    # r_star, m_star
    star_dict = {
        'Kepler-1627': [(0.881, 0.018), (0.953, 0.019)],
        'Kepler-1643': [(0.855, 0.044), (0.845, 0.025)],
        'KOI-7368': [(0.876, 0.035), (0.879, 0.018)],
        'KOI-7913': [(0.790, 0.049), (0.760, 0.025)],
    }

    #_r_pl = np.array([1, 2, 3, 4, 6, 8, 16])*u.R_earth
    _r_pl = np.array([16, 8, 6, 4, 3, 2, 1])*u.R_earth
    _pl_orbper = [2, 4, 8, 16, 32, 64]

    for _star_id in list(star_dict.keys()):

        for r_pl, pl_orbper in product(_r_pl, _pl_orbper):

            lcdir = os.path.join(PHOTDIR, _star_id)
            if not os.path.exists(lcdir): os.mkdir(lcdir)

            lcc = get_kepler_lightcurve(_star_id, download_dir=lcdir)
            if _star_id == 'KOI-7913':
                lcc = [l for l in lcc if l.LABEL == 'KIC 8873450']
            time, flux, flux_err = mask_quality_and_stitch(lcc)

            # injection-related steps
            r_star = star_dict[_star_id][0][0]*u.Rsun
            rp_rs = (r_pl / r_star).cgs.value
            epoch = np.nanmin(time)+np.random.uniform(low=0,high=pl_orbper)
            depth = rp_rs**2
            inj_dict = {
                'period':pl_orbper,
                'epoch':epoch,
                'depth':depth
            }
            time, flux, _ = lcu.inject_transit_signal(time, flux, inj_dict)
            star_id = _star_id + f"-synth-P{pl_orbper}d-Rp{int(r_pl.value)}"

            dtr_dict = {
                'method':'best', 'break_tolerance':0.5, 'window_length':0.5
            }
            search_method = 'tls'
            cachepath = os.path.join(
                LOCALDIR, f"{star_id}_{search_method}_iterative_search.pkl"
            )

            R_star = star_dict[_star_id][0][0]
            R_star_min = star_dict[_star_id][0][0] - 4*star_dict[_star_id][0][1]
            R_star_max = star_dict[_star_id][0][0] + 4*star_dict[_star_id][0][1]
            M_star = star_dict[_star_id][1][0]
            M_star_min = star_dict[_star_id][1][0] - 4*star_dict[_star_id][1][1]
            M_star_max = star_dict[_star_id][1][0] + 4*star_dict[_star_id][1][1]

            outdicts = (
                detrend_and_iterative_tls(
                    star_id, time, flux, dtr_dict,
                    period_min=1, period_max=100,
                    R_star=R_star, M_star=M_star,
                    R_star_min=R_star_min, R_star_max=R_star_max,
                    M_star_min=M_star_min, M_star_max=M_star_max,
                    n_transits_min=10,
                    search_method=search_method, n_threads=nworkers,
                    magisflux=True, cachepath=cachepath,
                    slide_clip_lo=4, slide_clip_hi=3,
                    verbose=True
                )
            )

            outdir = os.path.join(RESULTSDIR, 'cepher_koi_injectrecovery')
            if not os.path.exists(outdir): os.mkdir(outdir)

            dtr_stages_dict = outdicts['dtr_stages_dict']
            plot_detrend_check(
                star_id, outdir, dtr_dict,
                dtr_stages_dict, r=outdicts[0]['r'], instrument='kepler'
            )

            plot_iterative_planet_finding_results(
                star_id, search_method, outdir, cachepath, dtr_dict,
                overwrite=True
            )

            #TODO: add some log file of whether the recovery succeeded!
            # with open(cachepath, 'rb') as f:
            #     d = pickle.load(f)


if __name__ == "__main__":
    main()
