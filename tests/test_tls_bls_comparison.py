import numpy as np, matplotlib.pyplot as plt, pandas as pd
import batman
import pickle, os
from transitleastsquares import transitleastsquares
from astropy.timeseries import BoxLeastSquares
from astrobase.periodbase import kbls
from aeons.paths import DATADIR
import multiprocessing as mp
n_threads = mp.cpu_count()

pklpath = os.path.join(DATADIR, 'tests', 'KOI-7368_planetsearch_dtrcache.pkl')

# make synthetic signal

#for npoints in [1e3, 3e3, 5e3, 7e3, 1e4, 2e4, 4e4, 5e4]:
for npoints in [5e3, 5e4]:

    with open(pklpath, 'rb') as f:
        d = pickle.load(f)
    time, flux = d['search_time'], d['search_flux']
    time = time[:int(npoints)]
    flux = flux[:int(npoints)]

    # Use batman to create transits
    ma = batman.TransitParams()
    ma.t0 = 0.42  # time of inferior conjunction; first transit is X days after start
    ma.per = 7.202  # orbital period
    rp = 2e-2
    ma.rp = rp  # planet radius (in units of stellar radii)
    ma.a = 17.6  # semi-major axis (in units of stellar radii)
    ma.inc = 90  # orbital inclination (in degrees)
    ma.ecc = 0  # eccentricity
    ma.w = 90  # longitude of periastron (in degrees)
    ma.u = [0.27, 0.41]  # limb darkening coefficients
    ma.limb_dark = "quadratic"  # limb darkening model
    m = batman.TransitModel(ma, time)  # initializes model
    synthetic_signal = m.light_curve(ma) - 1  # calculates light curve

    flux += synthetic_signal

    df = pd.DataFrame({
        'time': time,
        'flux': flux
    })
    df.to_csv(f'KOI-7368_npoints{int(npoints)}.csv', index=False)

    #plt.figure(figsize=(14,4))
    #plt.scatter(time, flux)
    #plt.savefig('temp.png')

    oversampling_factor = 5
    cachepath = os.path.join(
        DATADIR, 'tests',
        f'KOI-7368_abtls_rprs{rp:.3f}_os{oversampling_factor}_npoints{int(npoints)}.pkl'
    )

    if not os.path.exists(cachepath):

        do_TLS = 1
        do_astropyBLS = 0
        do_astrobaseBLS = 1

        print(f'Beginning rp/rs={rp}...')

        if do_TLS:
            model = transitleastsquares(time, flux, verbose=True)
            results = model.power(use_threads=n_threads, show_progress_bar=True,
                                  R_star=0.88, R_star_min=0.85, R_star_max=0.91,
                                  M_star=0.88, M_star_min=0.85, M_star_max=0.91,
                                  period_min=5, period_max=10,
                                  n_transits_min=10, transit_template='default',
                                  transit_depth_min=10e-6,
                                  oversampling_factor=oversampling_factor)
        else:
            results = None

        if do_astropyBLS:
            # try astropy box_least_squares
            bls = BoxLeastSquares(time, flux)
            durations = np.array([0.5, 1, 2, 4])/24
            bls_results = bls.autopower(durations, oversample=10, minimum_period=5,
                                        maximum_period=10, minimum_n_transit=10)
            bls_period = bls_results.period[np.argmax(bls_results.power)]
        else:
            bls_results = None
            bls_period = None

        if do_astrobaseBLS:
            blsdict = kbls.bls_parallel_pfind(
                time, flux, 1e-4*flux, magsarefluxes=True, startp=5,
                endp=10, mintransitduration=0.01, maxtransitduration=0.12,
                nphasebins=200, autofreq=True, nbestpeaks=5,
                nworkers=n_threads, get_stats=True
            )
        else:
            blsdict = None


        outdict = {
            'tls_results': results,
            'bls_results': bls_results,
            'bls_period': bls_period,
            'ab_bls_results': blsdict,
            'time':time,
            'flux':flux,
        }
        with open(cachepath, 'wb') as f:
            pickle.dump(outdict, f)
            print(f"Wrote {cachepath}")

    else:
        with open(cachepath, 'rb') as f:
            td = pickle.load(f)
        print(f"Found {cachepath}, continue")
        print(f"TLS P: {td['tls_results']['period']}, "
              f"astropy BLS P: {td['bls_period']}, "
              f"astrobase BLS P: {td['ab_bls_results']['bestperiod']}"
             )

        plt.close('all')
        plt.plot(td['ab_bls_results']['periods'],
                 td['ab_bls_results']['lspvals'], lw=0.5)
        plt.savefig(f'KOI-7368_abbls_rprs{rp:.3f}_os{oversampling_factor}_npoints{int(npoints)}.png')
