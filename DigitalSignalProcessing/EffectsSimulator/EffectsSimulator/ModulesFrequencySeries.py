"""
EffectsEmmulatorPython
ModulesFrequencySeries.py
Landon Buell
14 Feb 2021
"""

            #### IMPORTS ####

import os
import sys

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

from ModulesGeneric import *

            #### MODULE DEFINITIONS ####

class NBandEquilizer(AbstractParentModule) :
    """
    NBandEquilize - 
        Parent Class of all N-band Equilizers
    --------------------------------
    (See AbstractParentModule for documentation)

    _nBands (int) : Number of filterbands to use in EQ
    _bandTypes (list_ 
    --------------------------------
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 nBands=1,bandTypes=[None]):
        """ Constructor for NBandEquilizer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._nBands = nBands
        self._bandTypes = bandTypes





