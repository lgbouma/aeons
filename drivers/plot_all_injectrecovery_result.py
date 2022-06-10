import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os
from glob import glob
from aeons.paths import RESULTSDIR

# results from rotation period measurements, via
# run_cepher_rotation_measurement -> plot_rotation_diagnostics
csvpath = os.path.join(
    RESULTSDIR, 'cepher_rotation_diagnostics',
    'tab_supp_CepHer_X_Kepler_ls_periods.csv'
)
df = pd.read_csv(csvpath)

csvpaths = glob(os.path.join(RESULTSDIR, 'cepher_all_injectrecovery',
                             '*-synth-*result.csv'))

star_ids = np.unique(
    [os.path.basename(c).split("-synth")[0] for c in csvpaths]
)

rdfs = []

for star_id in star_ids:

    csvpaths = glob(os.path.join(RESULTSDIR, 'cepher_all_injectrecovery',
                                 f'{star_id}-synth-*result.csv'))
    assert len(csvpaths) > 0

    rdf = pd.concat((pd.read_csv(f) for f in csvpaths))
    rdfs.append(rdf)

    rdf['recov'] = rdf.recovered | rdf.partial_recovered

    plt.close('all')
    f, ax = plt.subplots(figsize=(4,3))
    ax.scatter(
        rdf.period, rdf.r_pl, c=rdf.recov, s=5, zorder=1
    )

    _G = float(df.loc[df.star_id==star_id, 'g'])
    _bpmrp = float(df.loc[df.star_id==star_id, 'bpmrp'])
    _Prot = float(df.loc[df.star_id==star_id, 'ls_period'])
    a_90_10 = float(df.loc[df.star_id==star_id, 'a_90_10'])

    title = f'{star_id}: G={_G:.1f}, BP-RP={_bpmrp:.1f}, Prot={_Prot:.2f}, amp={a_90_10:.3f}'
    ax.set_title(title, fontsize='x-small')
    ax.update({
        'xlabel': 'P [days]',
        'ylabel': 'Rp [earths]',
        'xscale': 'log',
        'yscale': 'log',
        'xlim': [1,100],
        'ylim':[0.5,20]
    })
    outpath = os.path.join(RESULTSDIR, 'cepher_all_injectrecovery',
                            f'{star_id}-tls_recovery_result.png')
    f.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Made {outpath}')

mdf = pd.concat(rdfs)

plt.close('all')
f, ax = plt.subplots(figsize=(4,3))
ax.scatter(
    mdf.period, mdf.r_pl, c=mdf.recov, s=5, zorder=1
)

#_G = float(df.loc[df.star_id==star_id, 'g'])
#_Prot = float(df.loc[df.star_id==star_id, 'ls_period'])
#a_90_10 = float(df.loc[df.star_id==star_id, 'a_90_10'])
#title = f'{star_id}: G={_G}, Prot={_Prot}, a_90_10={a_90_10}'

ax.update({
    'xlabel': 'P [days]',
    'ylabel': 'Rp [earths]',
    #'title': title,
    'xscale': 'log',
    'yscale': 'log'
})
outpath = os.path.join(RESULTSDIR, 'cepher_all_injectrecovery',
                        f'merged-tls_recovery_result.png')
f.savefig(outpath, dpi=300, bbox_inches='tight')
print(f'Made {outpath}')


import IPython; IPython.embed()


