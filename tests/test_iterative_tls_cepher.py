"""
Q1: Can you find KOI-7368, and the second (injected) planet, iteratively?
"""
import os, pickle
import numpy as np, matplotlib.pyplot as plt, pandas as pd

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

def main(inject_synthetic=False):

    star_ids = [
        'Kepler-1627', 'Kepler-1643', 'KOI-7368', 'KOI-7913'
    ]

    for star_id in star_ids:

        lcdir = os.path.join(PHOTDIR, star_id)
        if not os.path.exists(lcdir): os.mkdir(lcdir)

        lcc = get_kepler_lightcurve(star_id, download_dir=lcdir)
        if star_id == 'KOI-7913':
            lcc = [l for l in lcc if l.LABEL == 'KIC 8873450']
        time, flux, flux_err = mask_quality_and_stitch(lcc)

        if inject_synthetic:
            inj_dict = {
                'period':7.202,
                'epoch':np.nanmin(time)+0.42,
                'depth':1000e-6
            }
            time, flux, _ = lcu.inject_transit_signal(time, flux, inj_dict)
            star_id = star_id + "-synth"

        dtr_dict = {
            'method':'best', 'break_tolerance':0.5, 'window_length':0.5
        }
        search_method = 'tls'
        cachepath = os.path.join(
            LOCALDIR, f"{star_id}_{search_method}_iterative_search.pkl"
        )

        outdicts = (
            detrend_and_iterative_tls(
                star_id, time, flux, dtr_dict,
                period_min=1, period_max=50,
                R_star_min=0.7, R_star_max=1,
                M_star_min=0.7, M_star_max=1,
                n_transits_min=3,
                search_method=search_method, n_threads=nworkers,
                magisflux=True, cachepath=cachepath,
                slide_clip_lo=4, slide_clip_hi=3,
                verbose=True
            )
        )

        outdir = os.path.join(RESULTSDIR, 'test_iterative_tls_cepher')
        if not os.path.exists(outdir): os.mkdir(outdir)

        dtr_stages_dict = outdicts['dtr_stages_dict']
        plot_detrend_check(
            star_id, outdir, dtr_dict,
            dtr_stages_dict, r=outdicts[0]['r'], instrument='kepler'
        )

        plot_iterative_planet_finding_results(
            star_id, search_method, outdir, cachepath, dtr_dict,
            overwrite=1
        )

        with open(cachepath, 'rb') as f:
            d = pickle.load(f)


if __name__ == "__main__":
    main()
