import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os
from glob import glob
from aeons.paths import RESULTSDIR

star_dict = {
    # Porb, Rp, Prot, Gaia G
    'Kepler-1643': [5.3, 2.32, 5.11, 13.8],
    'Kepler-1627': [7.2, 3.8, 2.64, 13.0],
    'KOI-7368': [6.84, 2.22, 2.61, 12.8],
    'KOI-7913': [24.3, 2.34, 3.39, 14.2],
}


for star_id in star_dict.keys():

    _Porb, _Rp, _Prot, _G = star_dict[star_id]

    csvpaths = glob(os.path.join(RESULTSDIR, 'cepher_koi_injectrecovery',
                                 f'{star_id}-synth*tls_recovery_result.csv'))
    assert len(csvpaths) > 0

    df = pd.concat((pd.read_csv(f) for f in csvpaths))

    df['recov'] = df.recovered | df.partial_recovered

    f, ax = plt.subplots(figsize=(4,3))
    ax.scatter(
        df.period, df.r_pl, c=df.recov, s=5, zorder=1
    )
    ax.scatter(
        _Porb, _Rp, s=10, marker='*', zorder=2
    )
    title = f'{star_id}: G={_G}, Prot={_Prot}'
    ax.update({
        'xlabel': 'P [days]',
        'ylabel': 'Rp [earths]',
        'title': title,
        'xscale': 'log',
        'yscale': 'log'
    })
    outpath = os.path.join(RESULTSDIR, 'cepher_koi_injectrecovery',
                            f'{star_id}-tls_recovery_result.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Made {outpath}')
