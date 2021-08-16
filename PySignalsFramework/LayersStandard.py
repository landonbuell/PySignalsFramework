"""
PySignalsFramework
LayersStandard.py
Landon Buell
5 Feb 2021
"""

            #### IMPORTS ####

import os
import sys

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

import PySignalsFramework.AudioTools as AudioTools

            #### STANDARD LAYER DEFINITIONS ####

class AbstractLayer :
    """
    AbstractLayer Type
        Abstract Base Type for all Layer Classes
        Acts as node in double linked list LayerChain
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
        self._type = "AbstractLayer"            # type of this layer instance
        self._sampleRate = sampleRate           # sample rate of this layer
        self._chainIndex = None                 # index in layer chain
        self._next = None                       # temporarily set next
        self._prev = None                       # temporarily set prev
        self.coupleToNext(next)                 # connect to next Layer
        self.coupleToPrev(prev)                 # connect to prev Layer
        self._shapeInput = inputShape           # input signal Shape
        self._shapeOutput = inputShape          # output signal shape
        self._init = False               # T/F is chain has been initialzed
        self._signal = np.array([])             # output Signal

    def __del__(self):
        """ Destructor for AbstractLayer Parent Class """
        pass

    def deepCopy(self):
        """ Return Deep Copy of this Instance """
        newLayer = AbstractLayer(self._name,self._sampleRate,
                                 self._shapeInput,self._shapeOutput,
                                 None,None)
        newLayer.Next = self._next
        newLayer.Prev = self._prev
        return newLayer
  
    """ Public Interface """

    def initialize (self,inputShape,**kwargs):
        """ Initialize this layer for usage in chain """
        self.setInputShape(inputShape)
        self.setOutputShape(inputShape)
        self._signal = np.empty(self._shapeOutput)
        self._init = True
        return self

    def call (self,X):
        """ Call this Layer with inputs X """
        if self._init == False:      # Not Initialized
            errMsg = self.__str__() + " has not been initialized\n\t" + "Call <instance>.Initialize() before use"
            raise NotImplementedError(errMsg)       
        return X

    def describe(self,detail=1):
        """ Desribe This layer in desired detail """
        print("-"*128)
        print("Name: {0} Type: {1}".format(self._name,type(self)))
        return None

    def coupleToNext(self,otherLayer):
        """ Couple to next Layer """
        if otherLayer:
            self._next = otherLayer
            otherLayer._prev = self
        else:
            self._next = None
        self._init = False
        return self

    def coupleToPrev(self,otherLayer):
        """ Couple to Previous Layer """
        if otherLayer:
            self._prev = otherLayer
            otherLayer._next = self
        else:
            self._prev = None
        self._init = False
        return self

    """ Getter & Setter Methods """

    def getName(self):
        """ Get Name of this Layer """
        return self._name

    def getType(self):
        """ Get Type of this Layer """
        return self._type

    def getSampleRate(self):
        """ Get Sample Rate for this Layer """
        return self._sampleRate

    def setSampleRate(self,x):
        """ Set Sample Rate for this Layer """
        self._sampleRate = x
        return self

    def getIndex(self):
        """ Get this Layer's chain index """
        return self._chainIndex

    def setIndex(self,x):
        """ Set this Layer's chain index """
        self._chainIndex = x
        return self

    def next(self):
        """ Get Next Layer in Chain """
        return self._next

    def prev(self):
        """ Get Previous Layer in Chain """
        return self._prev

    def getInputShape(self):
        """ Get the input shape of this layer """
        return self._shapeInput

    def setInputShape(self,x):
        """ Set the input shape of this layer """
        self._shapeInput = x
        self._init = False
        return self

    def getOutputShape(self):
        """ Get The output shape of this layer """
        return self._shapeOutput

    def setOutputShape(self,x):
        """ Set the output shape of this layer """
        self._shapeOutput = x
        self._init = False
        return self

    def getInit(self):
        """ Get T/F is this Layer is Initialized """
        return self._init

    def setInit(self,x):
        """ Set T/F if this Layer is Initialized """
        self._init = x
        return self

    def getSignal(self):
        """ Return the Output Signal of this Layer """
        return self._signal

    """ Magic Methods """

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' @ Idx " + str(self._chainIndex)

class AmplitudeEnvelope(AbstractLayer):
    """

    """
    pass

class AnalysisFramesConstructor (AbstractLayer):
    """
    AnalysisFramesConstructor Type - 
        Decompose a 1D time-domain signal into a 2D Signal of
        Short-time analysis frames w/ optional head and tail padding
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
    _framesInUse (int) : Number of frames used by 
    _padTail (int) : Number of zeros to tail-pad each analysis frame
    _padHead (int) : Number of zeros to head-pad each analysis frame
    _frameSize (int) : Size of each frame, includes samples + padding
    --------------------------------
    Return instantiated AnalysisFramesConstructor Object 
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,percentOverlap=0.75,maxFrames=256,tailPad=1024,headPad=0):
        """ Constructor for AnalysisFamesConstructor Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "AnalysisFrameConstructor"
        self._samplesPerFrame = samplesPerFrame
        self._percentOverlap = percentOverlap
        self._overlapSamples = int(samplesPerFrame*(1-percentOverlap))
        self._maxFrames = maxFrames
        self._framesInUse = 0
        self._padTail = tailPad
        self._padHead = headPad
        self._frameSize = self._padHead + self._samplesPerFrame + self._padTail
        self._shapeOutput = (self._maxFrames,self._frameSize)

    def __del__(self):
        """ Destructor for AnalysisFamesConstructor Instance """
        pass
        
    """ Public Interface """

    def initialize (self,inputShape=None,**kwargs):
        """ Initialize this module for usage in chain """
        super().initialize(inputShape,**kwargs)

        # format output signal
        self._shapeOutput = (self._maxFrames,self._frameSize)
        self._signal = np.zeros(shape=self._shapeOutput,dtype=np.float32)
        self._framesInUse = 0
        self._initialized = True 
        return self

    def call(self, X):
        """ Call this Module with inputs X """
        X = super().call(X)
        self._framesInUse = 0
        self.signalToFrames(X)
        return self._signal

    """ Protected Interface """

    def signalToFrames(self,X):
        """ Convert signal X into analysis Frames """
        frameStartIndex = 0
        for i in range(self._maxFrames):
            frame = X[frameStartIndex:frameStartIndex+self._samplesPerFrame]
            if (frame.shape == (self._samplesPerFrame,)):
                # Frame has correct shape
                self._signal[i , self._padHead:-self._padTail] = frame
                self._framesInUse += 1
            else:
                # frame has incorrect shape
                self._signal[i , self._padHead:self._padHead + frame.shape[0]] = frame
                self._framesInUse += 1
                break
            frameStartIndex += self._overlapSamples
        return self

    def framesToSignal(self,X):
        """ Convert Analysis Frames X into 1D signal """
        frameStartIndex = 0
        for i in range(self._framesInUse):
            frame = X[i,self._padHead:self._padHead + self._samplesPerFrame]
            self._signal[frameStartIndex : frameStartIndex + self._samplesPerFrame] = frame
            frameStartIndex += self._sample_samplesPerFrame
        return self

    """ Getter & Setter Methods """

    def getFrameParams(self):
        """ Get Frame Construction Parameters """
        params = [  self._samplesPerFrame,  self._percentOverlap,   self._maxFrames,
                    self._framesInUse,      self._padTail,          self._padHead]
        return params

    def setFrameParams(self,x):
        """ Set Frame Construction Parameters """
        self._samplesPerFrame   = x[0]
        self._percentOverlap    = x[1]
        self._maxFrames         = x[2]
        self._framesInUse       = x[3]
        self._padTail           = x[4]
        self._padHead           = x[5]
        self._overlapSamples = int(x[0] * (1 - x[1] ))
        self._initialized = False
        return self

    def getMaxFrames(self):
        """ Get Maximum number of analysis frames """
        return self._maxFrames

    def setMaxFrames(self,x):
        """ Set the Maximimber of analysis frames """
        self._maxFrames = x
        self.initialize(self._shapeInput)
        return self


class AnalysisFramesDestructor (AnalysisFramesConstructor):
    """
    AnalysisFramesDestructor Layer Type - 
        Destruct 2D array of analysis frames into 1D input waveform
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
    _framesInUse (int) : Number of frames used by 
    _padTail (int) : Number of zeros to tail-pad each analysis frame
    _padHead (int) : Number of zeros to tail-pad each analysis frame   
    _frameSize (int) : Size of each frame, includes samples + padding
    --------------------------------
    Return instantiated AnalysisFramesDestructor Object 
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,percentOverlap=0.75,maxFrames=256,tailPad=1024,headPad=0,
                 deconstructParams=None):
        """ Constructor for AnalysisFamesDestructor Instance """
        #super().__init__(name,sampleRate,inputShape,next,prev)
        super().__init__(name,sampleRate,inputShape,next,prev,
                         samplesPerFrame,percentOverlap,maxFrames,tailPad,headPad)
        self._type = "AnalysisFrameConstructor"
        if deconstructParams:           # Parameters from AnalysisFramesConstructor
            self.setFrameParams(deconstructParams)
        else:
            self._samplesPerFrame = samplesPerFrame
            self._percentOverlap = percentOverlap
            self._overlapSamples = int(samplesPerFrame*(1-percentOverlap))
            self._maxFrames = maxFrames
            self._padTail = tailPad
            self._padHead = headPad
            self._framesInUse = 0
        self._frameSize = self._padHead + self._samplesPerFrame + self._padTail

    """ Public Interface """

    def initialize (self,inputShape=None,**kwargs):
        """ Initialize this module for usage in chain """
        super().initialize(inputShape,**kwargs)

        # format output shape
        self._shapeOutput = (self._sample_samplesPerFrame * self._framesInUse,)
        self._signal = np.zeros(shape=self._shapeOutput,dtype=np.float32)
        self._framesInUse = 0
        self._initialized = True 
        return self

    def call(self, X):
        """ Call this Module with inputs X """
        X = super().call(X)
        self.framesToSignal(X)
        return self._signal

    """ Protected Interface """

class CustomCallable (AbstractLayer):
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
        """ Constructor for CustomCallable Layer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "CustomCallable"
        if callableFunction:
            self._callable = callableFunction
        else:
            raise ValueError("Must Provide callable argument for CustomCallable")
        self._callArgs = callableArgs
    
    def call(self,X):
        """ Call this Layer with inputs X """
        super().call(X)
        np.copyto(self.signal,self._callable(X,self._callArgs))
        return self._signal

class DiscreteFourierTransform(AbstractLayer):
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

    _freqAxis (arr) : Frequency Space Axis values
    --------------------------------
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for DiscreteFourierTransform Layer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)   
        self._freqAxis = np.array([])

    """ Public Interface """

    def initialize(self,inputShape=None,**kwargs):
        """ initialize Current Layer """
        super().initialize(inputShape,**kwargs)     
        self._signal = self._signal.astype('complex64')
        sampleSpacing = 1 / self._sampleRate
        self._freqAxis = fftpack.fftfreq(self._shapeOutput[-1],sampleSpacing)
        return self

    def call(self,X):
        """ Call this Layer w/ Inputs X """
        super().call(X)
        self.transform(X)
        return self._signal

    """ Protected Interface """

    def transform(self,X):
        """ Execute Discrete Fourier Transform on Signal X """
        nSamples = X.shape[-1]
        X = fftpack.fft(X,n=nSamples,axis=-1,overwrite_x=True)
        np.copyto(self._signal,X)
        return self

    """ Getter and Setter Methods """

    def getFreqAxis(self):
        """ Get the X-Axis Data """
        return self._freqAxis

    def setFreqAxis(self,x):
        """ Set the X-Axis Data """
        self._freqAxis = x
        return self

class DiscreteInvFourierTransform(AbstractLayer):
    """
    DiscreteInvFourierTransform - 
        Apply Inverse Discrete Fourier Transform to input signal
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
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for DiscreteInvFourierTransform Layer Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)

    def initialize(self,inputShape=None,**kwargs):
        """ Initialize Current Layer """
        super().initialize(self,inputShape,**kwargs)
        return self

    def transform(self,X):
        """ Execute Discrete Fourier Transform on Signal X """
        nSamples = X.shape[-1]
        X = fftpack.ifft(X,n=nSamples,axis=-1,overwrite_x=True)
        np.copyto(self._signal,X)
        return self

    def call(self,X):
        """ Call this Layer w/ Inputs X """
        super().call(X)
        self.transform(X)
        return self._signal

class Equilizer(AbstractLayer) :
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

        self._frequencyResponse = self.buildResponse()


    def buildResponse(self):
        """ Build this module's frequency response curve """
        return self

    def applyResponse(self,X):
        """ Apply Frequency response curve to the signal """
        return X

    def call(self,X):
        """ Call this Module with inputs X """
        super().call(X)

        return X

class IdentityLayer (AbstractLayer):
    """
    IdentityLayer Type - 
        Provides no Transformation of input, serves as placeholder layer if needed       
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

class IOLayer (AbstractLayer):
    """
   IOLayer Type - 
        Holds Input/Output Signals For Processing
        Commonly used as Head/Tail nodes in LayerChain
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
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "IOLayer"

    # Methods

    def initialize (self,inputShape=None,**kwargs):
        """ Initialize this layer for usage in chain """
        super().initialize(inputShape,**kwargs)
        # Initialize input Layer?
        return self

    def call (self,X):
        """ Call this Layer with inputs X """
        return super().call(X)

class LoggerLayer(AbstractLayer):
    """

    """
    pass

class ScaleAmplitudeLayer(AbstractLayer):
    """
    PlotSignal -
        Plot 1D or 2D signal in Time or Frequncy Space.
        Optionally show figure to console and/or save to specified directory
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

    _const (float) : Min/Max of signal will be this value
    _scaleFactor (float) : Value Required to scale amplitude to desire values
    --------------------------------
    """
    def __init__(self, name, sampleRate=44100, inputShape=None, next=None, prev=None,
                 normFactor=1):
        """ Initialize NormalizeLayer class Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._const = const
        self._scaleFactor = 0

    """ Public Interface """

    def initialize(self, inputShape, **kwargs):
        """ Initialize This Layer """
        super().initialize(inputShape, **kwargs)
        
        self._initialized = True
        return self

    def call(self,X):
        """ Call this Layer with Inputs X """
        super().call(X)
        self._signal = self.normalizeSignal(X,self._const)
        return self._signal

    """ Protected Interface """

    def normalizeSignal(signal,const=1):
        """ Normalize Signal to have max/min amplitde of +/- const """
        maxAmp = np.max(signal)
        minAmp = np.min(signal)
        if (maxAmp > np.abs(minAmp)):
            self._scaleFactor  = (const / maxAmp)
        else:
            self._scaleFactor = (-const / minAmp)
        signal = signal * self._scaleFactor
        return signal

    """ Getter & Setter Methods """

    def getNormFactor(self):
        """ Get Current Normalization Factor """
        return self._normFact
    
    def setNormFactor(self,x):
        """ Set Normalization Factor """
        self._normFactor = x
        return self

class PlotSignal(AbstractLayer):
    """
    PlotSignal -
        Plot 1D or 2D signal in Time or Frequncy Space.
        Optionally show figure to console and/or save to specified directory
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
    _saveFigure (bool) : If True, figures is saved to local drive
    self._xAxis (arr) : Data to use for x Axis
    --------------------------------
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 figurePath=None,figureName=None,show=False,save=True,xAxis=None):
        """ Constructor for AbstractLayer Parent Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "PlotSignal1D"

        if figurePath:
            self._figurePath = figurePath           
        else:
            self._figurePath = os.getcwd()
        if figureName:
            self._figureName = figureName
        else:
            self._figureName = self._name + "SignalPlot"

        self._showFigure = show
        self._saveFigure = save

        if xAxis is not None:
            self._xAxis = xAxis
        else:
            self._xAxis = np.array([])

    """ Public Interface """

    def initialize(self, inputShape, **kwargs):
        """ initialize this Layer for Usage """
        super().initialize(inputShape, **kwargs)
        if 'xAxis' in kwargs:
            self._xAxis = kwargs['xAxis']
        else:
            self._xAxis = np.arange(self._shapeInput[-1]) / self._sampleRate
        return self

    def call(self,X):
        """ Call current layer with inputs X """
        super().call(X)
        self._signal = X
        figureExportPath = os.path.join(self._figurePath,self._figureName)

        if (self._signal.dtype == np.complex64):
            data = np.array([self._signal.real,self._signal.imag],dtype=np.float32).transpose()
            AudioTools.Plotting.PlotGeneric(self._xAxis,data,labels=['real','imaginary'],
                            save=self._saveFigure,show=self._showFigure)

        else:
            AudioTools.Plotting.PlotGeneric(self._xAxis,self._signal,labels=['signal'],
                            save=self._saveFigure,show=self._showFigure)

        return self._signal
    
    """ Getter & Setter Methods """

    def getShowStatus(self):
        """ Get T/F if figure is shown to console """
        return self._showFigure

    def setShowStatus(self,x):
        """ Set T/F if figure is shown to console """
        self._showFigure = x
        return self

    def getSaveStatus(self):
        """ Get T/F if Figure is saved to local Path """
        return self._saveFigure

    def setSaveStatus(self,x):
        """ Set T/F if figure is saved to local Path """
        self._saveFigure = x
        return self

    def getXAxisData(self):
        """ Get the X-Axis Data """
        return self._xAxis

    def setXAxisData(self,x):
        """ Set the X-Axis Data """
        self._xAxis = x
        return self

class PlotSpectrogram (PlotSignal):
    """
    PlotSpectrogram -
        Plot 2D spectrogram as color-coded heat map in Time and Frequncy Space.
        Optionally show figure to console and/or save to specified directory
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
    _showFigure (bool) : If True, figure is displayed to console
    _saveFigure (bool) : If True, figure is saved to local drive
    _axisTime (arr) : Array of Values for time axis
    _axisFreq (arr) : Array of Values for frequency ax
    _logScale (bool) : If True, values are plotted on log scale
    _colorMap (str) : String indicating color map to use

    --------------------------------
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 timeAxis=np.array([]),freqAxis=np.array([]),logScale=True,colorMap='viridis',
                 figurePath=None,figureName=None,show=False,save=True):
        """ Constructor for PlotSpectrogram Instance """
        super().__init__(name,sampleRate,inputShape,next,prev,
                         figurePath,figureName,show,save,None)
        self._axisTime = timeAxis
        self._axisFreq = freqAxis
        self._logScale = logScale
        self._colorMap = colorMap
        raise NotImplementedType()
        
    """ Public Interface """

    def initialize(self, inputShape, **kwargs):
        """ Initialize This Layer """
        super().initialize(inputShape, **kwargs)

       
class ResampleLayer (AbstractLayer):
    """
    Resample Layer -
        Resample Signal to Desired Sample Rate
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
    def __init__(self,name,sampleRateNew,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for ResampleLayer instance """
        super().__init__(name,ssampleRate,inputShape,next,prev)
        self._sampleRateNew = sampleRateNew
        raise NotImplementedType()

class WindowFunction (AbstractLayer):
    """
    WindowFunction Type - 
         Apply a specified Window function (or callable) to a
         1D or 2D signal
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
    
    _nSamples (int) : Number of samples that the window is applied to
    _padTail (int) : Number of zeros to tail pad the window with
    _padHead (int) : Number of zeros to head pad the window with
    _frameSize (int) : Size of each frame, includes samples + padding

    _function (call/arr) : ???
    _window (arr) : Array of window function and padding
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 function=None,nSamples=1024,tailPad=1024,headPad=0):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "WindowFunctionLayer"
        
        self._nSamples = nSamples
        self._padTail = tailPad
        self._padHead = headPad
        self._frameSize = self._padHead + self._nSamples + self._padTail
        
        self._function = function;
        self._window = np.zeros(shape=self._frameSize)
        self._window[self._padHead:-self._padTail] = self._function(self._nSamples)

    def initialize(self,inputShape=None,**kwargs):
        """ Initialize Layer for Usage """
        super().initialize(inputShape,**kwargs)
        self._isInit = True
        return self

    def call(self,X):
        """ Call this module with inputs X """
        X = super().call(X)
        np.multiply(X,self._window,out=X)
        return X

    def setDeconstructionParams(self,params):
        """ Return a List of params to deconstruct Frames into Signal """
        self._samplesPerFrame = params[0]
        self._padTail = params[4]
        self._padHead = params[5]
        return self
