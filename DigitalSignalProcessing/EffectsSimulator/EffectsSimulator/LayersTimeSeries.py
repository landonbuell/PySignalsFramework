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

from LayersGeneric import *
from AudioTools import WindowFunctions

            #### MODULE DEFINITIONS ####

class InputLayer (AbstractParentLayer):
    """
    ModuleAnalysisFrames Type - 
        Deconstruct a 1D time-domain signal into a 2D array of overlapping analysis frames
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
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for AnalysisFames Instance """
        if not inputShape:
            raise ValueError("input shape must be provdied (cannot be \'None\')")
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "InputLayer"

    # Methods

    def Initialize (self,inputShape=None,**kwargs):
        """ Initialize this layer for usage in chain """
        super().Initialize(inputShape=None,**kwargs)
        # Initialize input Layer?
        return self

    def Call (self,X):
        """ Call this Layer with inputs X """
        super().Call(X)
        assert (x.ndim == 1)
        return X

class AnalysisFamesConstructor (AbstractParentLayer):
    """
    ModuleAnalysisFrames Type - 
        Construct 2D array of analysis frames from 1D input waveform
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

    def Initialize (self,inputShape=None,**kwargs):
        """ Initialize this module for usage in chain """
        super().Initialize(inputShape=None,**kwargs)
        # Get the input Shape
        if inputShape:
            self.SetInputShape(inputShape)
        else:
            self.SetInputShape(self.Prev.GetOutputShape)
        # format output shape
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

class ResampleLayer (AbstractParentLayer):
    """

    """
    pass

class WindowFunction (AbstractParentLayer):
    """
    WindowFunction Type - 
         Apply a window finction
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
    
    _windowType (str) : String indicating window function type
    _window(callable/array) : Callable
    _windowSize (int) : Size of window function in sample    
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 window=None,windowType=None,nSamples=None):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "WindowFunctionLayer"
        
        self._windowType = windowType
        self._windowSize = 0
        self._window = np.array([])

    def Initialize(self,inputShape=None,**kwargs):
        super().Initialize(inputShape=None,**kwargs)
        self._windowSize = self._shapeInput[-1]
        return self

    def Call(self,X):
        """ Call this module with inputs X """
        X = super().Call(X)
        X = np.multiply(X,self._window,out=X)
        return X


class AmplitudeEnvelope(AbstractParentLayer):
    """

    """
    pass

