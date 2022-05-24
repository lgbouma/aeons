"""
Q1: Can you find Kepler-1627, Kepler-1643, KOI-7913, and KOI-7368?
Q2: How sensitive is this search?
"""
import os, pickle
import numpy as np, matplotlib.pyplot as plt
from aeons.paths import DATADIR, PHOTDIR, LOCALDIR, RESULTSDIR

import multiprocessing as mp
nworkers = mp.cpu_count()

from cdips.lcproc.find_planets import (
    detrend_and_iterative_tls, plot_detrend_check,
    plot_planet_finding_results
)

def main():

    csvpath = os.path.join(DATADIR, 'tests', 'KOI-7368_npoints50000.csv')
    df = pd.read_csv(csvpath)
    time, flux = np.array(df.time), np.array(df.flux)

    dtr_method, break_tolerance, window_length = 'best', 0.5, 0.5
    dtr_dict = {'method':dtr_method,
                'break_tolerance':break_tolerance,
                'window_length':window_length}

    search_method = 'tls'
    cachepath = os.path.join(
        LOCALDIR, f"{star_id}_{search_method}_iterative_search.pkl"
    )
    outdicts = (
        detrend_and_iterative_tls(
            star_id, time, flux, dtr_dict,
            period_min=5, period_max=10,
            R_star_min=0.8, R_star_max=1,
            M_star_min=0.8, M_star_max=1,
            n_transits_min=3,
            search_method=search_method, n_threads=nworkers,
            magisflux=True, return_extras=True, cachepath=cachepath,
            slide_clip_lo=4, slide_clip_hi=3
            verbose=True
        )
    )

    outdir = os.path.join(RESULTSDIR, 'test_iterative_tls_cepher')
    if not os.path.exists(outdir): os.mkdir(outdir)

    dtr_stages_dict = outdicts['dtr_stages_dict']
    plot_detrend_check(
        star_id, outdir, dtr_dict,
        dtr_stages_dict, r=r, instrument='kepler'
    )

    plot_iterative_planet_finding_results(
        star_id, search_method, outdir, cachepath, dtr_dict
    )

    with open(cachepath, 'rb') as f:
        d = pickle.load(f)


if __name__ == "__main__":
    main()
