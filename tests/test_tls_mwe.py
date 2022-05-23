import pandas as pd, numpy as np, matplotlib.pyplot as plt
from transitleastsquares import transitleastsquares
import multiprocessing as mp
n_threads = mp.cpu_count()

csvpath = 'KOI-7368_npoints50000.csv'
df = pd.read_csv(csvpath)
time, flux = np.array(df.time), np.array(df.flux)

from astropy.stats import sigma_clip
flux = sigma_clip(flux, sigma_upper=3, sigma_lower=3)

stime = time[~flux.mask]
sflux = flux[~flux.mask]

model = transitleastsquares(stime, sflux, verbose=True)
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

model2 = transitleastsquares(t_second_run, y_second_run, verbose=True)
results2 = model2.power(
    use_threads=n_threads, show_progress_bar=True, R_star=0.88,
    R_star_min=0.85, R_star_max=0.91, M_star=0.88, M_star_min=0.85,
    M_star_max=0.91, period_min=5, period_max=10, n_transits_min=10,
    transit_template='default', transit_depth_min=10e-6, oversampling_factor=5
)
# raises the following warning:
# /home/lbouma/miniconda3/envs/py37/lib/python3.7/site-packages/transitleastsquares/main.py:205:
# UserWarning: No transit were fit. Try smaller "transit_depth_min"

print(results['period'])
# gives nan! (since the chi^2 = 46,409 == the number of data points for all
# proposed periods)

plt.close("all")
plt.plot(results2['periods'], results2['power'], lw=0.5)
plt.xlabel('period [days]')
plt.ylabel('power')
plt.savefig('tls_mwe_sigclip_run2.png', dpi=400)

