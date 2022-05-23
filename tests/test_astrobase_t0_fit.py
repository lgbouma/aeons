"""
The purpose of this test was to provide a path toward implementing a
working t0-finder for astrobase.periodbase.kbls.

The **period** finding from this routine works very well.

The **epoch** finding routine (hilariously) does not.

The data at /data/tests/kepler1627_example.csv would provide one way of testing
this -- using the known preferred depth, duration etc for that planet, this
test would ideally show that _get_bls_stats is capable of retrieving a correct
epoch.
"""
from astrobase.periodbase.kbls import _get_bls_stats

raise NotImplementedError('never wrote the test!')

_get_bls_stats(stimes, smags, serrs, thistransdepth, thistransduration,
               ingressdurationfraction, nphasebins, thistransingressbin,
               thistransegressbin, thisbestperiod, thisnphasebins,
               magsarefluxes=magsarefluxes, verbose=verbose)
