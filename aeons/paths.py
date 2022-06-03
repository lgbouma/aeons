import os, socket
from aeons import __path__

DATADIR = os.path.join(os.path.dirname(__path__[0]), 'data')
RESULTSDIR = os.path.join(os.path.dirname(__path__[0]), 'results')
TABLEDIR = os.path.join(RESULTSDIR, 'tables')
PHOTDIR = os.path.join(DATADIR, 'phot')
PAPERDIR = os.path.join(os.path.dirname(__path__[0]), 'paper')
DRIVERDIR = os.path.join(os.path.dirname(__path__[0]), 'drivers')

#
# NOTE: assumes ~/local/ exists.  you may wish to point this to a symlink
# somewhere else.
#
LOCALDIR = os.path.join(os.path.expanduser('~'), 'local', 'aeons')

dirlist = [DATADIR, RESULTSDIR, TABLEDIR, PHOTDIR, PAPERDIR, DRIVERDIR,
           LOCALDIR]

for d in dirlist:
    if not os.path.exists(d):
        os.mkdir(d)
        print(f'Made {d}')
