"""
Landon Buell
EffectsSimulator
LayersGeneric
5 Feb 2020
"""

            #### IMPORTS ####

import os
import sys

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

import AudioTools

            #### LAYER DEFINITIONS ####

class AbstractParentLayer :
    """
    AbstractLayer Type
        Abstract Base Type for all Layer Classes
        Acts as node in double linked list
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
    Abstract class - Make no instance
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for AbstractLayer Parent Class """
        self._name = name                       # name of this layer instance
        self._type = "AbstractParentLayer"      # type of this layer instance
        self._sampleRate = sampleRate           # sample rate of this layer
        self._chainIndex = None                 # index in layer chain
        self._next = next                       # next layer in chain
        self._prev = prev                       # previous layer in chain
        self._shapeInput = inputShape           # input signal Shape
        self._shapeOutput = inputShape          # output signal shape
        self._initialized = False               # T/F is chain has been initialzed
        self._signal = np.array([])             # output Signal
                             
    # Methods

    def Initialize (self,inputShape=None,**kwargs):
        """ Initialize this layer for usage in chain """
        if inputShape:
            self._shapeInput = inputShape
            self._shapeOutput = inputShape
        else:
            self._shapeInput = (1,)
            self._shapeOutput = (1,)
        self._signal = np.empty(shape=self.GetOutputShape)
        self._initialized = True 
        return self

    def Call (self,X):
        """ Call this Layer with inputs X """
        if self._initialized == False:
            errMsg = self.__str__() + " has not been initialized\n\t" + "Call Instance.Initialize() before use"
            raise NotImplementedError(errMsg)
        return X

    @property
    def Next(self):
        """ Return the next Layer in chain """
        return self._next

    @property
    def Prev(self):
        """ Return the previous Layer in chain """
        return self._prev

    # Local Properties

    @property
    def GetSampleRate(self):
        """ Get the Sample Rate for this Layer """
        return self._sampleRate

    def SetSampleRate(self,x):
        """ Set the Samplke Rate for this Layer """
        self._sampleRate = x
        return self

    @property
    def GetIndex(self):
        """ Get this Layer's chain index """
        return self._chainIndex
    
    def SetIndex(self,x):
        """ Set this Layer's chain index """
        self._chainIndex = x
        return self

    @property
    def InputShape(self):
        """ Get the input shape of this layer """
        return self._shapeInput

    def SetInputShape(self,x):
        """ Set the input shape of this layer """
        self._shapeInput = x
        return self

    @property
    def OutputShape(self):
        """ Get The output shape of this layer """
        return self._shapeOutput

    def SetOutputShape(self,x):
        """ Set the output shape of this layer """
        self._shapeOutput = x
        return self

    # Magic Methods

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' @ " + str(self._chainIndex)

class IdentityLayer (AbstractParentLayer):
    """
    IdentityLayer Type - Provides no Transformation of input
        Serves as head/tail nodes of FX chain Graph
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
    Return Instantiated identityLayer
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for AbstractLayer Parent Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "IdentityLayer"

class CustomCallable (AbstractParentLayer):
    """
    CustomCallable Type - Returns User defined transformation 
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

    _callable (callable) : User-denfined or desired function transformation
    _callArgs (list) : List of arguments to pass to callable function
    --------------------------------
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 callableFunction=None,callableArgs=[]):
        """ Constructor for CustomCallable Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "CustomCallable"
        if callableFunction:
            self._callable = callableFunction
        else:
            raise ValueError("Must Provide callable argument for CustomCallable")
        self._callArgs = callableArgs
    
    def Call(self,X):
        """ Call this Layer with inputs X """
        super().Call(X)
        return self._callable(X,self._callArgs)

class PlotSignal1D(AbstractParentLayer):
    """
    CustomCallable Type - Returns User defined transformation 
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

    _savePath (str) : Local path where plot of 1D signal is exported to
    _figureName (str) : Name to save local plots
    _showFigure (bool) : If True, figures is displayed to console
    --------------------------------
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 figurePath=None,figureName=None,show=False):
        """ Constructor for AbstractLayer Parent Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "PlotSignal1"

        if figurePath:
            self._figurePath = figurePath           
        else:
            self._figurePath = os.getcwd()
        if figureName:
            self._figureName = figureName
        else:
            self._figureName = "Signal1D"
        self._showFigure = show

    def Call(self,X):
        """ Call current layer with inputs X """
        super().Call(X)
        figureExportPath = os.path.join(self._figurePath,self._figureName)
        cntr = 0
        while True:
            if os.path.isfile(figureExportPath):
                # the file already exists
                figureExportPath += str(cntr)
            else:
                figureExportPath += str(cntr)
        xData = np.arange(X.shape) / self._sampleRate
        AudioTools.Plotting.PlotGeneric(xData,X,
                                        save=figureExportPath,show=self._show)
        return X
    
    @property
    def GetShowStatus(self):
        """ Get if figure is shown to console """
        return self._showFigure

    def SetShowStatus(self,x):
        """ Set if figure is shown to console """
        self._showFigure = x
        return self

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

