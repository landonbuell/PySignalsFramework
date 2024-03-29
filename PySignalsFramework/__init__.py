"""
PySignalsFramework
__init__.py
Landon Buell
14 June 2021
"""

        #### IMPORTS ####

__version__ = "Indevelopment"

__doc__ = \
"""
NAME:   PySignalsFramework

DESCRIPTION: Python Framework of Digital Signal Processing Tools

CONTENTS:

    Namespace: AdminTools

    Namespace: AudioTools

    Namespace: Effects Systems

    Namespace: LayerChain

    Namespace: LayersCustom

    Namespace: LayersStandard

"""

# IMPORT NUMPY
try:
    import numpy as np
    NUMPY = True
except ImportError:   
    NUMPY = False

# IMPORT MATPLOTLIB
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

# IMPORT SCIPY.SIGNAL
try:
    import scipy.signal as scisig
    SCIPY_SIGNAL = True
except ImportError:
    SCIPY_SIGNAL = False

# IMPORT SCIPY.IO.WAVFILE
try:
    import scipy.io.wavfile
    SCIPY_WAVFILE = True
except ImportError:
    SCIPY_WAVFILE = False
