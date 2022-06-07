import numpy as np, matplotlib.pyplot as plt, pandas as pd
import os, pickle
from copy import deepcopy
from glob import glob

from aeons.paths import DATADIR, RESULTSDIR, LOCALDIR
from cdips.utils.gaiaqueries import given_source_ids_get_gaia_data

import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

csvpath = os.path.join(DATADIR, 'tables', 'tab_supp_CepHer_X_Kepler.csv')
df = pd.read_csv(csvpath)

## usually would query gaia archive to get the stellar info.  however
## (2022/06/07) it's down for maintenance this week.
# gdf = given_source_ids_get_gaia_data(
#     np.array(df.dr3_source_id.astype(np.int64)),
#     'tab_supp_CepHer_X_Kepler',
#     overwrite=False,
#     gaia_datarelease='gaiaedr3'
# )

csvpath = os.path.join(DATADIR, 'tables',
                       '20220311_Kerr_CepHer_Extended_Candidates_v0-result.csv')
gdf = pd.read_csv(csvpath)
_ = np.intersect1d(df.dr3_source_id, gdf.source_id)
assert len(_) == len(df)
mdf = df.merge(gdf, left_on='dr3_source_id', right_on='source_id', how='left')
assert len(mdf) == len(df)

df = deepcopy(mdf)

outdir = os.path.join(RESULTSDIR, 'cepher_rotation_diagnostics')
if not os.path.exists(outdir): os.mkdir(outdir)

#
# get relevant periodogram data
#
outrows = []

for ix, r in df.iterrows():

    kep_id = str(r['kepid'])
    star_id = f'KIC-{kep_id}'

    print(f"Collecting {star_id}...")
    pklpath = os.path.join(
        LOCALDIR, f"{star_id}_tls_iterative_search_dtrcache.pkl"
    )
    assert os.path.exists(pklpath)

    with open(pklpath, 'rb') as f:
        d = pickle.load(f)

    if 'a_90_10' not in d:
        flux = d['dtr_stages_dict']['flux']
        a_90_10 = np.nanpercentile(flux, 90) - np.nanpercentile(flux, 10)
        d['a_90_10'] = a_90_10

    ls_period = np.round(d['dtr_stages_dict']['lsp_dict']['ls_period'], 6)
    ls_fap = d['dtr_stages_dict']['lsp_dict']['ls_fap']
    ls_amplitude = d['dtr_stages_dict']['lsp_dict']['ls_amplitude']
    a_90_10 = np.round(d['a_90_10'], 6)
    bpmrp = np.round(r['phot_bp_mean_mag'] - r['phot_rp_mean_mag'], 4)
    g = np.round(r['phot_g_mean_mag'], 4)
    M_G = np.round(r['phot_g_mean_mag'] + 5*np.log10(r['parallax']/1e3) + 5, 4)
    weight = np.round(r['weight'], 6)

    outrow = (
        star_id, ls_period, ls_fap, ls_amplitude, a_90_10, bpmrp, g, M_G,
        weight, str(r['dr2_source_id']), str(r['dr3_source_id'])
    )
    outrows.append(outrow)

df = pd.DataFrame(
    outrows,
    columns=['star_id', 'ls_period', 'ls_fap', 'ls_amplitude', 'a_90_10',
             'bpmrp', 'g', 'M_G', 'weight', 'dr2_source_id', 'dr3_source_id']
)
# We want LS periods for everything!
assert not np.any(np.isnan(df.ls_period))

from cdips.gyroage import PleiadesInterpModel
sel = (df.ls_period < PleiadesInterpModel(df.bpmrp, bounds_error=False))
sel_p1 = (df.ls_period < PleiadesInterpModel(df.bpmrp, bounds_error=False)+1)
sel_p2 = (df.ls_period < PleiadesInterpModel(df.bpmrp, bounds_error=False)+2)

df['subpleiades'] = sel
df['subpleiades_p1'] = sel_p1
df['subpleiades_p2'] = sel_p2

# Either you're within 1 day above the Pleiades (i.e., <~120 myr,
# gyrochronologically), or you need to be an F-star with a strong weight.
# (The latter class is obviously less interesting for finding small planets)
sel_fn = (
    sel_p1
    |
    ((df.bpmrp<0.5) & (df.weight>0.2))
)
df['selected'] = sel_fn

outcsv = os.path.join(
    RESULTSDIR, 'cepher_rotation_diagnostics',
    'tab_supp_CepHer_X_Kepler_ls_periods.csv'
)
df.to_csv(outcsv, index=False)
print(f"Wrote {outcsv}")

xytuples = [
    # x, y, xscale, yscale
    ('bpmrp', 'ls_period', 'linear', 'linear'),
    ('bpmrp', 'ls_period', 'linear', 'log'),
    ('bpmrp', 'ls_amplitude', 'linear', 'log'),
    ('bpmrp', 'a_90_10', 'linear', 'log'),
    ('bpmrp', 'M_G', 'linear', 'linear')
]

for xytuple in xytuples:

    xkey, ykey, xscale, yscale = xytuple

    plt.close('all')
    outpath = os.path.join(
        RESULTSDIR, 'cepher_rotation_diagnostics',
        f'{xkey}_vs_{yscale}_{ykey}.png'
    )

    fig, ax = plt.subplots(figsize=(4,3))

    color = np.log10(df['weight'])
    cmap = mpl.cm.get_cmap('plasma')

    _p = ax.scatter(
        df[xkey], df[ykey], s=3, c=color, zorder=2, cmap=cmap, vmax=-1
    )

    if xkey == 'bpmrp' and ykey =='ls_period':
        interp_bpmrp = np.linspace(0,3,int(1e3))
        prot_interp = PleiadesInterpModel(interp_bpmrp, bounds_error=False)
        ax.plot(interp_bpmrp, prot_interp, zorder=1, lw=1, label='pleiades')
        ax.plot(interp_bpmrp, prot_interp+1, zorder=1, lw=1, label='pleiades+1')
        ax.plot(interp_bpmrp, prot_interp+2, zorder=1, lw=1, label='pleiades+2')

    axins1 = inset_axes(ax, width="3%", height="20%",
                        loc='lower right', borderpad=0.7)

    cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                      extend="max")
    cb.ax.tick_params(labelsize='x-small')
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.set_label('$\log_{10}$ weight', fontsize='xx-small')

    if xkey == 'bpmrp' and ykey =='ls_period':
        ax.legend(loc='upper right', fontsize='xx-small')

    ax.update({
        'xlabel': xkey,
        'ylabel': ykey,
        'xscale': xscale,
        'yscale': yscale
    })

    if ykey == 'M_G':
        ax.set_ylim(ax.get_ylim()[::-1])

    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Made {outpath}')
    plt.close('all')


for xytuple in xytuples:

    xkey, ykey, xscale, yscale = xytuple

    plt.close('all')
    outpath = os.path.join(
        RESULTSDIR, 'cepher_rotation_diagnostics',
        f'sel_{xkey}_vs_{yscale}_{ykey}.png'
    )

    fig, ax = plt.subplots(figsize=(4,3))

    selcolor = np.log10(df[df.selected]['weight'])
    cmap = mpl.cm.get_cmap('plasma')

    _p = ax.scatter(
        df[df.selected][xkey], df[df.selected][ykey], s=3, c=selcolor, zorder=3,
        cmap=cmap, vmax=-1
    )

    if xkey == 'bpmrp' and ykey =='ls_period':
        interp_bpmrp = np.linspace(0,3,int(1e3))
        prot_interp = PleiadesInterpModel(interp_bpmrp, bounds_error=False)
        ax.plot(interp_bpmrp, prot_interp, zorder=1, lw=1, label='pleiades')
        ax.plot(interp_bpmrp, prot_interp+1, zorder=1, lw=1, label='pleiades+1')
        ax.plot(interp_bpmrp, prot_interp+2, zorder=1, lw=1, label='pleiades+2')

    axins1 = inset_axes(ax, width="3%", height="20%",
                        loc='lower right', borderpad=0.7)

    cb = fig.colorbar(_p, cax=axins1, orientation="vertical",
                      extend="max")
    cb.ax.tick_params(labelsize='x-small')
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.set_label('$\log_{10}$ weight', fontsize='xx-small')

    if xkey == 'bpmrp' and ykey =='ls_period':
        ax.legend(loc='upper right', fontsize='xx-small')

    ax.update({
        'xlabel': xkey,
        'ylabel': ykey,
        'xscale': xscale,
        'yscale': yscale
    })

    if ykey == 'M_G':
        ax.set_ylim(ax.get_ylim()[::-1])

    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f'Made {outpath}')
    plt.close('all')

N = len(df[df.selected])
print(f'Proceeding with {N} stars based on rotation periods (or weight, for BP-RP<0.5).')
print(f"These stars are at {outcsv}")
