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

from LayersGeneric import *

            #### MODULE DEFINITIONS ####

class DiscreteFourierTransform(AbstractParentLayer):
    """
    DiscreteFourierTransform - 
        Apply Discrete Fourier Transform to input signal
    --------------------------------
    _name (str) : Name for user-level identification
    _type (str) : Type of Layer Instance
    _sampleRate (int) : Number of samples per second  
    _chainIndex (int) : Location where Layer sits in chain 
    _next (AbstractParentLayer) : Next layer in the layer chain
    _prev (AbstractParentLayer) : Prev layer in the layer chain  
    _shapeInput (tup) : Indicates shape (and rank) of layer input
    _shapeOutput (tup) : Indicates shape (and rank) of layer output
    _initialized (bool) : Indicates if Layer has been initialized    
    _signal (arr) : Signal from Transform  

    _bands (list[bands]) : Bands to apply to this equilizer
    _nBands (int) : Number of filterbands to use in EQ
    --------------------------------
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for NBandEquilizer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)

    def Initialize(self,*args):
        """ Initialize Current Layer """
        super().Initialize()
        return self

    def Call(self,X):
        """ Call this Layer w/ Inputs X """
        super().Call(X)
        return X

class Equilizer(AbstractParentLayer) :
    """
    NBandEquilizer - 
        Parent Class of all N-band Equilizers
    --------------------------------
    _name (str) : Name for user-level identification
    _type (str) : Type of Layer Instance
    _sampleRate (int) : Number of samples per second  
    _chainIndex (int) : Location where Layer sits in chain 
    _next (AbstractParentLayer) : Next layer in the layer chain
    _prev (AbstractParentLayer) : Prev layer in the layer chain  
    _shapeInput (tup) : Indicates shape (and rank) of layer input
    _shapeOutput (tup) : Indicates shape (and rank) of layer output
    _initialized (bool) : Indicates if Layer has been initialized    
    _signal (arr) : Signal from Transform  

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

