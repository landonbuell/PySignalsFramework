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

            #### MODULE DEFINITIONS ####

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
