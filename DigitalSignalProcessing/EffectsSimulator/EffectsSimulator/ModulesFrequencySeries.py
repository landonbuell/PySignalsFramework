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

class Equilizer(AbstractParentModule) :
    """
    NBandEquilize - 
        Parent Class of all N-band Equilizers
    --------------------------------
    (See AbstractParentModule for documentation)

    _bands (list[bands]) : Bands to apply to this equilizer
    _nBands (int) : Number of filterbands to use in EQ
    --------------------------------
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 bands=[]):
        """ Constructor for NBandEquilizer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "Equilizier"

        self._bands = bands
        self._nBands = len(bands)

        self._frequencyResponse = self.BuildResponse()


    def BuildResponse(self):
        """ Build this module's frequency response curve """

        return self

    def ApplyResponse(self,X):
        """ Apply Frequency response curve to the signal """

        return X

    def Call(self,X):
        """ Call this Module with inputs X """
        super().Call(X)

        return X






