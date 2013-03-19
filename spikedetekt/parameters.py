'''
Loads default parameters and puts them in a dictionary which should be used
for access to parameters across modules.

e.g.

def somefunc():
    FPC = Parameters['FPC']
    ...

or to load several, do something like:

    N_CH, S_TOTAL, FPC = eval('(N_CH, S_TOTAL, FPC)', Parameters)
    
'''
from defaultparameters import *

__all__ = ['Parameters', 'GlobalVariables']

Parameters = dict((k, v) for k, v in globals().iteritems() if not k.startswith('_'))
GlobalVariables = {
    'warnings':0,
    'log_fd':None,
    }

if __name__=='__main__':
    print GlobalParameters.keys()
    