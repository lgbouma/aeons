import numpy as np, matplotlib.pyplot as plt

def sixtyeight(flux):
    """
    Calculate the 68th percentile of the distribution of the flux.
    """
    med_flux = np.nanmedian(flux)

    up_sixtyeight = (
        np.nanpercentile( flux-med_flux, 84 )
        -
        np.nanpercentile( flux-med_flux, 50 )
    )
    lo_sixtyeight = (
        np.nanpercentile( flux-med_flux, 50 )
        -
        np.nanpercentile( flux-med_flux, 16 )
    )

    sixtyeight = np.mean([up_sixtyeight, lo_sixtyeight])

    return sixtyeight

def p2p_rms(flux):
    """
    Calculate the 68th percentile of the distribution of the residuals from the
    median value of δF_i = F_{i} - F_{i+1}, where i is an index over time.
    """
    dflux = np.diff(flux)
    med_dflux = np.nanmedian(dflux)

    up_p2p = (
        np.nanpercentile( dflux-med_dflux, 84 )
        -
        np.nanpercentile( dflux-med_dflux, 50 )
    )
    lo_p2p = (
        np.nanpercentile( dflux-med_dflux, 50 )
        -
        np.nanpercentile( dflux-med_dflux, 16 )
    )

    p2p = np.mean([up_p2p, lo_p2p])

    return p2p

for seed in [42, 3141]:

    print(42*'=')
    print(f"Seed {seed}")

    np.random.seed(seed)
    n_points = 1000
    scale = 0.05

    err = np.random.normal(loc=0, scale=scale, size=n_points)
    y = np.ones(n_points) + err

    print(f"Gaussian errors (σ={scale}):")
    print(f"P2P RMS: {p2p_rms(y):.4f}, STDEV: {np.std(y):.4f}, 68 percentile: {sixtyeight(y):.4f}")

    outlier_ind = np.unique(np.random.randint(0, n_points, size=20))
    outlier_val = np.random.uniform(low=-0.1, high=0.1, size=len(outlier_ind))

    y[outlier_ind] = outlier_val

    print("Gaussian errors + 2% outliers:")
    print(f"P2P RMS: {p2p_rms(y):.4f}, STDEV: {np.std(y):.4f}, 68 percentile: {sixtyeight(y):.4f}")

    # sample 20 points per period
    time = np.linspace(0, n_points/20, n_points)
    sine = 0.2*np.sin(time)
    y += sine

    print("Gaussian errors + 2% outliers + sine:")
    print(f"P2P RMS: {p2p_rms(y):.4f}, STDEV: {np.std(y):.4f}, 68 percentile: {sixtyeight(y):.4f}")

    plt.close("all")
    fig, axs = plt.subplots(nrows=2)
    axs[0].scatter(time, y)
    axs[1].scatter(time[1:], np.diff(y))
    axs[0].set_ylabel('flux')
    axs[1].set_ylabel('diff(flux)')
    axs[1].set_xlabel('time')
    plt.savefig(f"lc_seed{seed}.png", dpi=400)
