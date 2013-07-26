from __future__ import with_statement
import os, sys, numpy

scripts = ["scripts/detektspikes.py",
           ]

from distutils.core import setup, Extension
    
setup(name="spikedetekt",
      scripts=scripts,
      version="0.2 beta",
      author="Shabnam N. Kadir, Cyrille Rossant, Dan F.G. Goodman, John Schulman, Kenneth D. Harris",
      author_email="kenneth@cortexlab.net",
      description="Spike sorting for multi-site probes",
      license="GPL3",
      url="https://github.com/klusta-team/spikedetekt",
      packages=["spikedetekt"],
      )
