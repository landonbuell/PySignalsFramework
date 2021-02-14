"""
EffectsEmmulatorPython
ModulesTimeSeries.py
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

class AnalysisFames (AbstractParentModule):
    """
    ModuleAnalysisFrames Type - 
        Deconstruct a 1D time-domain signal into a 2D array of overlapping analysis frames
    --------------------------------
    (See AbstractParentModule for documentation)
    _samplesPerFrame (int) : Number of samples used in each analysisFrame
    _overlapSamples (int) : Number of samples 
    _maxFrames (int) : Maximum number of analysis frames to use
    _zeroPad (int) : Number of zeros to pad each analysis frame
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,overlapSamples=768,maxFrames=256,zeroPad=1024):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._samplesPerFrame = samplesPerFrame
        self._overlapSamples = overlapSamples
        self._maxFrames = maxFrames
        self._zeroPad = self._zeroPad

    def Initialize (self):
        """ Initialize this module for usage in chain """
        super().Initialize()
        self._shapeOutput = (self._samplesPerFrame + self._zeroPad,
                            self._maxFrames)
        self._signal = np.zeros(shape=self._shapeOutput,dtype=np.float32)
        self._initialized = True 
        return self

    def SignalToFrames(self,X):
        """ Convert signal X into analysis Frames """
        for i in range(len(self._maxFrames)):
            self._signal[0:self._samplesPerFrame] = frame
        return self

    def Call(self, X):
        """ Call this Module with inputs X """
        super().Call(X)


        return self._signal


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


