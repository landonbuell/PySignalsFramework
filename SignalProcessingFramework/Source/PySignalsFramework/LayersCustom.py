"""
Landon Buell
EffectsSimulator
LayersStandard
5 Feb 2020
"""

            #### IMPORTS ####

import os
import sys

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack

import PySignalsFramework.AudioTools as AudioTools
import PySignalsFramework.LayersStandard as Layers

            #### CUSTOM LAYER DEFINITIONS ####

class BasicDelay(Layers.AbstractLayer):
    """
    BasicDelay Layer Type
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

    _decayEnvelope (func) : Function to model decay envelope
    --------------------------------
    Abstract class - Make no instance
    """
    pass

class SigmoidOverdrive(Layers.AbstractLayer):
    """
    SigmoidOverdrive Layer Type
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

    _overdriveFunction (func) : function to transform Signal
    --------------------------------
    Abstract class - Make no instance
    """

    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None):
        """ Constuctor for BasicOverdrive class Instance """
        self._type = "BasicOverdrive"  

    """ Public Interface """

    def Initialize(self, inputShape, **kwargs):
        """ Initialize This Layer Instance """
        return super().Initialize(inputShape, **kwargs)

    def Call(self,X):
        """ Call this Layer with Inputs X """
        super().Call(X)
        self._signal = self.SigmoidDistortion(X)
        return self._signal

    """ Protected Interface """

    def SigmoidDistortion(self,X):
        """ Sigmoid Clipping to Signal """
        b = 2 * np.max(np.abs(X))
        c = 1.0
        Y = b*((1 / (1 + np.exp(-c*X))) - 0.5 )
        return Y



    

