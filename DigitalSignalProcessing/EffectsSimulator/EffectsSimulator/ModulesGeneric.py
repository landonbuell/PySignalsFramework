"""
Landon Buell
EffectsEmmulatorPython
Modules
5 Feb 2020
"""

            #### IMPORTS ####

import os
import sys

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

            #### MODULE DEFINITIONS ####

class AbstractParentModule :
    """
    AbstractModule Type
        Abstract Base Type for all Module Classes
        Acts as node in double linked list
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
    --------------------------------
    Abstract class - Make no instance
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for AbstractModule Parent Class """
        self._name = name
        self._type = "AbstractParentModule"

        self._next = next
        self._prev = prev

        self._chainIndex = None
        self._initialized = False

        self._shapeInput = inputShape
        self._shapeOutput = inputShape

        self._signal = np.array([])  
        self._sampleRate = sampleRate                 
        

    # Local Properties

    @property
    def InputShape(self):
        """ Get the input shape of this module """
        return self._shapeInput

    @property
    def OutputShape(self):
        """ Get The output shape of this module """
        return self._shapeOutput

    def SetInputShape(self,x):
        """ Set the input shape of this module """
        self._shapeInput = x
        return self

    def SetOutputShape(self,x):
        """ Set the output shape of this module """
        self._shapeOutput = x
        return self

    # Methods

    def Initialize (self,*args):
        """ Initialize this module for usage in chain """
        try:
            self._shapeInput = self._prev._shapeOutput
        except:
            self._shapeInput = (1,)
        self._initialized = True 
        return self

    def Call (self,X):
        """ Call this Module with inputs X """
        if self._initalized == False:
            errMsg = self.__str__() + " has not been initialized\n\t" + "Call Instance.Initialize() before use"
            raise NotImplementedError(errMsg)
        return X

    # Magic Methods

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + " " + self._name + " @ " + str(self._chainIndex)

class IdentityModule (AbstractParentModule):
    """
    IdentityModule Type - Provides no Transformation of input
        Serves as head/tail nodes of FX chain Graph
    --------------------------------
    (See AbstractParentModule for documentation)
    --------------------------------
    Return Instantiated identityModule
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constructor for AbstractModule Parent Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "IdentityModule"

class CustomCallableModule (AbstractParentModule):
    """
    CustomCallable Type - Returns User defined transformation 
    --------------------------------
    (See AbstractParentModule for documentation)
    _call (callable) : User-denfined or desired function transformation
    --------------------------------
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 callableFunction=None):
        """ Constructor for AbstractModule Parent Class """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "CustomCallable"
        if callableFunction:
            self._call = callableFunction
        else:
            raise ValueError("Must Provide callable argument for CustomCallable")
    
    def Call(self,X):
        """ Call this Module with inputs X """
        super().Call(X)
        return self._call(X)
