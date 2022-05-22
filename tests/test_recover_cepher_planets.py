"""
Q1: Can you find Kepler-1627, Kepler-1643, KOI-7913, and KOI-7368?
Q2: How sensitive is this search?
"""
import os, pickle
import numpy as np, matplotlib.pyplot as plt
from aeons.getters import get_kepler_lightcurve
from aeons.processing import mask_quality_and_stitch
from aeons.paths import PHOTDIR, LOCALDIR, RESULTSDIR

import multiprocessing as mp
nworkers = mp.cpu_count()

from cdips.lcproc.find_planets import (
    run_periodograms_and_detrend, plot_detrend_check,
    plot_planet_finding_results
)

def main():

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

        dtr_method, break_tolerance, window_length = 'best', 0.5, 0.5
        dtr_dict = {'method':dtr_method,
                    'break_tolerance':break_tolerance,
                    'window_length':window_length}

        search_method = 'bls'
        cachepath = os.path.join(
            LOCALDIR, f"{star_id}_{search_method}_search.pkl"
        )
        r, search_time, search_flux, dtr_stages_dict = (
            run_periodograms_and_detrend(
                star_id, time, flux, dtr_dict,
                search_method=search_method, n_threads=nworkers,
                magisflux=True, return_extras=True, cachepath=cachepath,
                period_min=1, period_max=100, verbose=True
            )
        )

        outdir = os.path.join(RESULTSDIR, 'recover_cepher_planets')
        if not os.path.exists(outdir): os.mkdir(outdir)

        plot_detrend_check(
            star_id, outdir, dtr_dict,
            dtr_stages_dict, r=r, instrument='kepler'
        )

        plot_planet_finding_results(
            star_id, search_method, outdir, cachepath, dtr_dict
        )

        with open(cachepath, 'rb') as f:
            d = pickle.load(f)


if __name__ == "__main__":
    main()
