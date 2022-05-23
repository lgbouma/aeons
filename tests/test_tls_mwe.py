import pandas as pd, numpy as np, matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares
import multiprocessing as mp
n_threads = mp.cpu_count()

def p2p_rms(flux):
    dflux = np.diff(flux)
    med_dflux = np.nanmedian(dflux)
    up_p2p = (
        np.nanpercentile( np.sort(dflux-med_dflux), 84 )
        -
        np.nanpercentile( np.sort(dflux-med_dflux), 50 )
    )
    lo_p2p = (
        np.nanpercentile( np.sort(dflux-med_dflux), 50 )
        -
        np.nanpercentile( np.sort(dflux-med_dflux), 16 )
    )
    p2p = np.mean([up_p2p, lo_p2p])
    return p2p

csvpath = 'KOI-7368_npoints50000.csv'
df = pd.read_csv(csvpath)
time, flux = np.array(df.time), np.array(df.flux)

from astropy.stats import sigma_clip
flux = sigma_clip(flux, sigma_upper=3, sigma_lower=3)

stime = time[~flux.mask]
sflux = flux[~flux.mask]

dy = np.ones_like(stime)*p2p_rms(sflux)
model = transitleastsquares(stime, sflux, dy=dy, verbose=True)
results = model.power(
    use_threads=n_threads, show_progress_bar=True, R_star=0.88,
    R_star_min=0.85, R_star_max=0.91, M_star=0.88, M_star_min=0.85,
    M_star_max=0.91, period_min=5, period_max=10, n_transits_min=10,
    transit_template='default', transit_depth_min=10e-6, oversampling_factor=5
)

plt.close("all")
plt.plot(results['periods'], results['power'], lw=0.5)
plt.xlabel('period [days]')
plt.ylabel('power')
plt.savefig('tls_mwe_sigclip_run1.png', dpi=400)

from transitleastsquares import transit_mask, cleaned_array

intransit = transit_mask(stime, results.period, 2*results.duration, results.T0)

y_second_run = sflux[~intransit]
t_second_run = stime[~intransit]
t_second_run, y_second_run = cleaned_array(t_second_run, y_second_run)

dy = np.ones_like(t_second_run)*p2p_rms(y_second_run)
model2 = transitleastsquares(t_second_run, y_second_run, dy=dy, verbose=True)
results2 = model2.power(
    use_threads=n_threads, show_progress_bar=True, R_star=0.88,
    R_star_min=0.85, R_star_max=0.91, M_star=0.88, M_star_min=0.85,
    M_star_max=0.91, period_min=5, period_max=10, n_transits_min=10,
    transit_template='default', transit_depth_min=10e-6, oversampling_factor=5
)

print(results2['period'])

plt.close("all")
plt.plot(results2['periods'], results2['power'], lw=0.5)
plt.xlabel('period [days]')
plt.ylabel('power')
plt.savefig('tls_mwe_sigclip_run2.png', dpi=400)
