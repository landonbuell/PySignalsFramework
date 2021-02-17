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

class AnalysisFamesConstructor (AbstractParentModule):
    """
    ModuleAnalysisFrames Type - 
        Deconstruct a 1D time-domain signal into a 2D array of overlapping analysis frames
    --------------------------------
    _name (str) : Name for user-level identification
    _type (str) : Type of Module Instance
    _next (AbstractParentModule) : Next module in the module chain
    _prev (AbstractParentModule) : Prev module in the module chain
    _chainIndex (int) : Location where Module sits in chain 
    _initialized (bool) : Indicates if Module has been initialized    
    _shapeInput (tup) : Indicates shape (and rank) of module input
    _shapeOutput (tup) : Indicates shape (and rank) of module output
    _signal (arr) : Signal from Transform
    _sampleRate (int) : Number of samples per second 

    _samplesPerFrame (int) : Number of samples used in each analysisFrame
    _percentOverlap (float) : Indicates percentage overlap between adjacent frames [0,1)
    _overlapSamples (int) : Number of samples overlapping
    _maxFrames (int) : Maximum number of analysis frames to use
    _zeroPad (int) : Number of zeros to pad each analysis frame
    _framesInUser (int) : Number of frames used by 
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,percentOverlap=0.75,maxFrames=256,zeroPad=1024):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "AnalysisFrameConstructor"
        self._samplesPerFrame = samplesPerFrame
        self._percentOverlap = percentOverlap
        self._overlapSamples = int(samplesPerFrame*(1-percentOverlap))
        self._maxFrames = maxFrames
        self._zeroPad = zeroPad
        self._framesInUse = 0

    def Initialize (self):
        """ Initialize this module for usage in chain """
        super().Initialize()
        self._shapeOutput = (self._maxFrames,
                             self._samplesPerFrame + self._zeroPad)
        self._signal = np.zeros(shape=self._shapeOutput,dtype=np.float32)
        self._framesInUse = 0
        self._initialized = True 
        return self

    def SignalToFrames(self,X):
        """ Convert signal X into analysis Frames """
        frameStartIndex = 0
        for i in range(self._maxFrames):
            frame = X[frameStartIndex:frameStartIndex+self._samplesPerFrame]
            try:
                self._signal[i,0:self._samplesPerFrame] = frame
            except ValueError:
                self._signal[i,0:len(frame)] = frame
                break
            frameStartIndex += self._overlapSamples
            self._framesInUse += 1
        return self

    def Call(self, X):
        """ Call this Module with inputs X """
        X = super().Call(X)
        self.SignalToFrames(X)
        return self._signal

class Resample (AbstractParentModule):
    """

    """
    pass

class WindowFunction (AbstractParentModule):
    """
    WindowFunction Type - 
         Apply a window finction
    --------------------------------
    _name (str) : Name for user-level identification
    _type (str) : Type of Module Instance
    _next (AbstractParentModule) : Next module in the module chain
    _prev (AbstractParentModule) : Prev module in the module chain
    _chainIndex (int) : Location where Module sits in chain 
    _initialized (bool) : Indicates if Module has been initialized    
    _shapeInput (tup) : Indicates shape (and rank) of module input
    _shapeOutput (tup) : Indicates shape (and rank) of module output
    _signal (arr) : Signal from Transform
    _sampleRate (int) : Number of samples per second 
    
    _windowType (str) : String indicating window function type
    _windowCall (call) : Callable
    _windowSize (int) : Size of window function in sample    
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 windowType=None,windowSize=None):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "WindowFunctionModule"
        
        self._windowType = windowType
        self._windowSize = windowSize
        self._window = None

    def Initialize(self):
        super().Initialize()
        return self

    def Call(self,X):
        """ Call this module with inputs X """
        X = super().Call(X)
        raise NotImplementedError()
        return X


class AmplitudeEnvelope(AbstractParentModule):
    """

    """
    pass

