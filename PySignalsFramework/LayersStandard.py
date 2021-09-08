"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           LayersStandard.py
Description:
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
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 
    --------------------------------
    Abstract class - Make no instance
    """

    def __init__(self,name,type="AbstractLayer",inputShape=None,next=None,prev=None):
        """ Constructor for AbstractLayer Parent Class """
        self._name = name                       # name of this layer instance
        self._type = type                       # type of this layer instance
        self._shapeInput = inputShape           # input signal Shape
        self._shapeOutput = inputShape          # output signal shape
        self._next = None                       # temporarily set next
        self._prev = None                       # temporarily set prev
        self._isInit = False                    # T/F is chain has been initialzed
        
        self._signal = AudioTools.Signal(
            data=None,shape=inputShape)         # Place to Hold Signal 

        self.coupleToNext(next)                 # connect to next Layer
        self.coupleToPrev(prev)                 # connect to prev Layer
        
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

    def initialize (self,inputShape=None,**kwargs):
        """ Initialize this layer for usage in chain """
        if (inputShape is not None):
            self.setInputShape(inputShape)
            self.setOutputShape(inputShape)
        self._isInit = True
        return self

    def call (self,X):
        """ call this Layer with inputs X """
        if self._init == False:      # Not Initialized
            errMsg = self.__str__() + " has not been initialized\n\t" + "call <instance>.initialize() before use"
            raise NotImplementedError(errMsg)
        if (type(X) != AudioTools.Signal):
            X = AudioTools.Signal(X)        # cast to Signal Obj
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
        self._isInit = False
        return self

    def coupleToPrev(self,otherLayer):
        """ Couple to Previous Layer """
        if otherLayer:
            self._prev = otherLayer
            otherLayer._next = self
        else:
            self._prev = None
        self._isInit = False    
        return self

    """ Getter & Setter Methods """

    def getName(self):
        """ Get Name of this Layer """
        return self._name

    def getType(self):
        """ Get Type of this Layer """
        return self._type

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
        self._signal.setShape(x)
        self._init = False
        return self

    def getNext(self):
        """ Get Next Layer in Chain """
        return self._next

    def setNext(self,x):
        """ Set Next Layer in Chain """
        if (type(x) != AbstractLayer):
            raise TypeError("Next layer must be a sub-type of AbstractLayer")
        else:
            self._next = x
        return self

    def getPrev(self):
        """ Get Previous Layer in Chain """
        return self._prev

    def setPrev(self,x):
        """ Set Previous Layer in Chain """
        if (type(x) != AbstractLayer):
            raise TypeError("Prev layer must be a sub-type of AbstractLayer")
        else:
            self._prev = x
        return self

    def getInitStatus(self):
        """ Get T/F is this Layer is Initialized """
        return self._init

    def setInitStatus(self,x):
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
        return self._type + ": \'" + self._name

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
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 

    _frameParams    FrameParams         Parameters for Analysis Frames Construction
    --------------------------------
    Return instantiated AnalysisFramesConstructor Object 
    """
    def __init__(self,name,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,samplesOverlap=768,maxFrames=256,
                 tailPad=1024,headPad=0,initParams=None):
        """ Constructor for AnalysisFamesConstructor Instance """
        super().__init__(name,"AnalysisFrameConstructor",inputShape,next,prev)
        self._frameParams = None

        if (initParams is not None):
            # Use init Params as construction parameters
            self._frameParams = initParams.deepCopy()
        else:
            # Use other arguments as construction parameters
            self._frameParams = AudioTools.FrameParams(
            samplesPerFrame,samplesOverlap,maxFrames,tailPad,headPad)



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
        """ call this Module with inputs X """
        X = super().call(X)
        self._framesInUse = 0
        self.signalToFrames(X)
        return self._signal

    @property
    def frameParams(self):
        """ Return Refrence to _frameParams struct instance """
        return self._frameParams

    """ Protected Interface """

    def signalToFrames(self,X):
        """ Convert signal X into analysis Frames """
        frameStartIndex = 0
        frameEndIndex = frameStartIndex 
        for i in range(self._maxFrames):
            frameEndIndex += self._frameParams.getSamplesPerFrame()
            frame = X[frameStartIndex:frameEndIndex]
            if (frame.shape == (self._samplesPerFrame,)):
                # Frame has correct shape
                self._signal[i , self._padHead:-self._padTail] = frame
                self._framesInUse += 1
            else:
                # frame has incorrect shape
                self._signal[i , self._padHead:self._padHead + frame.shape[0]] = frame
                self._framesInUse += 1
                break
            frameStartIndex += self._frameParams.getSamplesOverlap()
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
        return self._frameParams

    def setFrameParams(self,x):
        """ Set Frame Construction Parameters """
        self._frameParams = x
        return self


class AnalysisFramesDestructor (AnalysisFramesConstructor):
    """
    AnalysisFramesDestructor Layer Type - 
        Destruct 2D array of analysis frames into 1D input waveform
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 

    _frameParams    FrameParams         Parameters for Analysis Frames Construction
    --------------------------------
    Return instantiated AnalysisFramesDestructor Object 
    """
    def __init__(self,name,inputShape=None,next=None,prev=None,
                 samplesPerFrame=1024,percentOverlap=0.75,maxFrames=256,
                 tailPad=1024,headPad=0,destParams=None):
        """ Constructor for AnalysisFamesDestructor Instance """
        super().__init__(name,"AnalysisFrameDestructor",inputShape,next,prev,
                         samplesPerFrame,percentOverlap,maxFrames,tailPad,headPad,destParams)

    def __del__(self):
        """ Destructor for AnalysisFamesDestructor Instance """
        pass

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
        """ call this Module with inputs X """
        X = super().call(X)
        self.framesToSignal(X)
        return self._signal

    """ Protected Interface """

class CustomCallable (AbstractLayer):
    """
    Customcallable Type - Returns User defined transformation 
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation  

    _callable       callable            User-denfined or desired function transformation
    _callArgs       list                List of arguments to pass to callable function
    --------------------------------
    """
    def __init__(self,name,inputShape=None,next=None,prev=None,
                 callableFunction=None,callableArgs=[]):
        """ Constructor for Customcallable Layer Instance """
        super().__init__(name,"CustomCallable",sampleRate,inputShape,next,prev)

        if callableFunction:
            self._callable = callableFunction
        else:
            raise ValueError("Must Provide callable argument for Customcallable")
        self._callArgs = callableArgs

    def __del__(self):
        """ Destructor for Customcallable Instance """
        pass

    """ Public Interface """
    
    def call(self,X):
        """ call this Layer with inputs X """
        super().call(X)
        raise NotImplementedError()
        return self._signal

class DiscreteFourierTransform(AbstractLayer):
    """
    DiscreteFourierTransform - 
        Apply Discrete Fourier Transform to input signal
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 
    --------------------------------
    """
    def __init__(self,name,inputShape=None,next=None,prev=None):
        """ Constructor for DiscreteFourierTransform Layer Instance """
        super().__init__(name,"DiscreteFourierTransform",inputShape,next,prev)   

    def __del__(self):
        """ Destructor for DiscreteFourierTransform Instance """
        pass

    """ Public Interface """

    def initialize(self,inputShape=None,**kwargs):
        """ initialize Current Layer """
        super().initialize(inputShape,**kwargs)     
        self._signal.setDataType('complex64')
        return self

    def call(self,X):
        """ call this Layer w/ Inputs X """
        super().call(X)
        self.transform(X.getData())
        self.setSignalDomain()
        return self._signal

    """ Protected Interface """

    def transform(self,arr):
        """ Execute Discrete Fourier Transform on Signal X """

        nSamples = arr.shape[-1]
        arr = fftpack.fft(arr,nSamples,axis=-1,overwrite_x=True)
        self._signal.setData(arr)
        return self

    def setSignalDomain(self):
        """ Set Signal Domain Based on Shape """
        if (self._signal._data.ndim == 1):
            # 1D Signal, TIME -> FREQ
            self._signal.setDomain("FREQ")
        elif (self._signal._data.ndim == 2):
            # 1D Signal, TIME -> FREQ
            self._signal.setDomain("BOTH")
        else:
            # Greater than 2D signal, raise error
            raise ValueError("Singal must be 1D or 2D!")
        return self

    """ Getter and Setter Methods """


class DiscreteInvFourierTransform(AbstractLayer):
    """
    DiscreteInvFourierTransform - 
        Apply Inverse Discrete Fourier Transform to input signal
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 
    --------------------------------
    """
    def __init__(self,name,inputShape=None,next=None,prev=None):
        """ Constructor for DiscreteInvFourierTransform Instance """
        super().__init__(name,"DiscreteInvFourierTransform",inputShape,next,prev)

    def __del__(self):
        """ Destructor for DiscreteInvFourierTransform Instance """
        pass

    """ Public Interface """

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
    Equilizer - 
        Parent Class of all N-band Equilizers
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation  

    _bands          list[bands]         Bands to apply to this equilizer
    --------------------------------
    """

    def __init__(self,name,inputShape=None,next=None,prev=None,
                 bands=[]):
        """ Constructor for NBandEquilizer Instance """
        super().__init__(name,"Equilizier",inputShape,next,prev)

        self._bands = bands
        self._nBands = len(bands)

        self._frequencyResponse = self.buildResponse()

    def __del__(self):
        """ Destructor for NBandEquilizer Instance """
        pass

    """ Public Interface """

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
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation  
    --------------------------------
    Return Instantiated identityLayer
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for IdentityLayer Instance """
        super().__init__(name,"IdentityLayer",inputShape,next,prev)

    def __del__(self):
        """ Destructor for IdentityLayer Instance """
        pass

class IOLayer (AbstractLayer):
    """
   IOLayer Type - 
        Holds Input/Output Signals For Processing
        Commonly used as Head/Tail nodes in LayerChain
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation   
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for IOLayer Instance """
        super().__init__(name,"IOLayer",inputShape,next,prev)

    def __del__(self):
        """ Destructor for IOLayer Instance """
        pass

    """ Public Interface """

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

    def __init__(self):
        """ Constructor for LoggerLayer Instance """
        pass

    def __del__(self):
        """ Destructor for LoggerLayer Instance """
        pass

class ScaleAmplitudeLayer(AbstractLayer):
    """
    PlotSignal -
        Plot 1D or 2D signal in Time or Frequncy Space.
        Optionally show figure to console and/or save to specified directory
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation  

    _const          float               Min/Max of signal will be this value
    _scaleFactor    float               Value Required to scale amplitude to desire values
    --------------------------------
    """
    def __init__(self,name,inputShape=None,next=None,prev=None,
                 normFactor=1):
        """ Constructor ScaleAmplitudeLayer Instance """
        super().__init__(name,"ScaleAmplitudeLayer",inputShape,next,prev)

        self._const = const
        self._scaleFactor = 0

    def __del__(self):
        """ Destructor for ScaleAmplitudeLayer Instance """
        pass

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
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 

    _savePath       str                 Local path where plot of 1D signal is exported to
    _figureName     str                 Name to save local plots
    _showFigure     bool                If True, figures is displayed to console
    _saveFigure     bool                If True, figures is saved to local drive
    _xAxis          arr[float]          Data to use for x Axis
    --------------------------------
    """

    def __init__(self,name,inputShape=None,next=None,prev=None,
                 figurePath=None,figureName=None,show=False,save=True,xAxis=None):
        """ Constructor for PlotSignal Parent Class """
        super().__init__(name,"PlotSignal",inputShape,next,prev)

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

    def __del__(self):
        """ Destructor for PlotSignal Instance """
        pass

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
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 

    _savePath       str                 Local path where plot of 1D signal is exported to
    _figureName     str                 Name to save local plots
    _showFigure     bool                If True, figures is displayed to console
    _saveFigure     bool                If True, figures is saved to local drive
    _axisTime       arr[float]          Array of Values for time axis
    _axisFreq       arr[float]          Array of Values for frequency ax
    _logScale       bool                If True, values are plotted on log scale
    _colorMap       str                 String indicating color map to use
    --------------------------------
    
    """
    def __init__(self,name,inputShape=None,next=None,prev=None,
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

    def __del__(self):
        """ Destructor for PlotSpectrogram Instance """
        pass
        
    """ Public Interface """

    def initialize(self, inputShape, **kwargs):
        """ Initialize This Layer """
        super().initialize(inputShape, **kwargs)

       
class ResampleLayer (AbstractLayer):
    """
    Resample Layer -
        Resample Signal to Desired Sample Rate
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation 
    --------------------------------
    Abstract class - Make no instance
    """
    def __init__(self,name,newSampleRate,inputShape=None,next=None,prev=None):
        """ Constructor for ResampleLayer instance """
        super().__init__(name,"ResampleLayer",inputShape,next,prev)
        self._newSampleRate = newSampleRate
        raise NotImplementedType()

    def __del__(self):
        """ ResampleLayer for PlotSpectrogram Instance """
        pass


class WindowFunction (AbstractLayer):
    """
    WindowFunction Type - 
         Apply a specified Window function (or callable) to a
         1D or 2D signal
    --------------------------------
    _name           string              Name for user-level identification
    _type           string              Type of Layer Instance
    _shapeInput     tuple[int]          Indicates shape (and rank) of layer input
    _shapeOutput    tuple[int]          Indicates shape (and rank) of layer output
    _next           AbstractLayer       Next layer in the layer chain
    _prev           AbstractLayer       Prev layer in the layer chain      
    _isInit         bool                Indicates if Layer has been initialized    
    _signal         Signal              Signal Resulting from transformation   
    
    _nSamples       int                 Number of samples that the window is applied to
    _padTail        int                 Number of zeros to tail pad the window with
    _padHead        int                 Number of zeros to head pad the window with
    _frameSize      int                 Size of each frame, includes samples + padding

    _function       str                 String to indicate window function to use
    _window         arr[float]          Array of window function and padding
    --------------------------------
    Return instantiated AnalysisFrames Object 
    """
    
    def __init__(self,name,inputShape=None,next=None,prev=None,
                 function=None,nSamples=1024,tailPad=1024,headPad=0):
        """ Constructor for AnalysisFames Instance """
        super().__init__(name,"WindowFunctionLayer",inputShape,next,prev)
        
        self._nSamples = nSamples
        self._padTail = tailPad
        self._padHead = headPad
        self._frameSize = self._padHead + self._nSamples + self._padTail
        
        self._function = function;
        self._window = np.zeros(shape=self._frameSize)
        self._window[self._padHead:-self._padTail] = self._function(self._nSamples)

    def __del__(self):
        """ ResampleLayer for AnalysisFames Instance """
        pass

    """ Public Interface """

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
