import pandas as pd, numpy as np
from transitleastsquares import transitleastsquares
import multiprocessing as mp
n_threads = mp.cpu_count()

csvpath = 'KOI-7368_npoints50000.csv'
df = pd.read_csv(csvpath)
time, flux = np.array(df.time), np.array(df.flux)

model = transitleastsquares(time, flux, verbose=True)
results = model.power(
    use_threads=n_threads, show_progress_bar=True, R_star=0.88,
    R_star_min=0.85, R_star_max=0.91, M_star=0.88, M_star_min=0.85,
    M_star_max=0.91, period_min=5, period_max=10, n_transits_min=10,
    transit_template='default', transit_depth_min=10e-6, oversampling_factor=5
)

# raises the following warning:
# /home/lbouma/miniconda3/envs/py37/lib/python3.7/site-packages/transitleastsquares/main.py:205:
# UserWarning: No transit were fit. Try smaller "transit_depth_min"

print(results['period'])
# gives nan! (since the chi^2 = 50,000 == the number of data points for all
# proposed periods)
