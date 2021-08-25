"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           LayersCustom.py
Description:
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

class ClipOverdriveLayer(Layers.AbstractLayer):
    """
    ClipDistortion Layer Type
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

    _thresholdBase (float) : percent threshold to begin clipping
    _thresholdCurr (float) : Actuall threshold to clip based on max/min amp   
    _softClip (bool) : If true, signal is clipped w/ log of value
    --------------------------------
    Abstract class - Make no instance
    """
    def __init__(self,name,sampleRate=44100,inputShape=None,next=None,prev=None,
                 threshold = 0.8, softClip = True):
        """ Constuctor for BasicOverdrive class Instance """
        super().__init__(name,sampleRate,inputShape,next,prev)
        self._type = "ClipOverdrive"  
        self._thresholdBase = threshold
        self._thresholdCurr = threshold
        self._softClip = softClip

    """ Public Interface """

    def initialize(self, inputShape, **kwargs):
        """ Initialize This Layer Instance """
        super().Initialize(inputShape, **kwargs)

        return self

    def call(self,X):
        """ Call this Layer with Inputs X """
        super().call(X)
        self._signal = np.copy(X)
        maxAmp = np.max(np.abs(self._signal))
        self.setThreshold( maxAmp * self._thresholdBase)
        # Apply Clipping Functions
        if self._softClip:
            self.softClipSignal()
        else:
            self.hardClipSignal()
        return self._signal

    """ Protected Interface """

    def hardClipSignal(self):
        """ Apply Hard Clipping to Signal """
        for i in range(self._signal.shape[-1]):
            currSample = self._signal[i]
            if (currSample > self._thresholdCurr):          # beat threshold (high)
                self._signal[i] = self._thresholdCurr       # scale by log
            elif (currSample < -self._thresholdCurr):       # beat threshold (low)
                self._signal[i] = -self._thresholdCurr      # scale by log
            else:
                continue
        return None

    def softClipSignal(self):
        """ Apply Soft Clipping to Signal """
        for i in range(self._signal.shape[-1]):
            X = self._signal[i]
            if (X > self._thresholdCurr):          # beat threshold (high)
                self._signal[i] = self._thresholdCurr/4 * (np.log(np.abs(X)) + 1)
            elif (X < -self._thresholdCurr):           # beat threshold (low)
                self._signal[i] = -self._thresholdCurr/4 * (np.log(np.abs(X)) + 1)
            else:
                continue
        return None

    """ Getter & Setter Methods """

    def setThreshold(self,x=None):
        """ Set the current Threshold for Clipping """
        self._thresholdCurr = x
        return self

    def getThreshold(self):
        """ get the current threshold for clipping """
        return self._thresholdCurr

class SigmoidOverdriveLayer(Layers.AbstractLayer):
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

    def initialize(self, inputShape, **kwargs):
        """ Initialize This Layer Instance """
        return super().Initialize(inputShape, **kwargs)

    def call(self,X):
        """ call this Layer with Inputs X """
        super().call(X)
        self._signal = self.sigmoidDistortion(X)
        return self._signal

    """ Protected Interface """

    def sigmoidDistortion(self,X):
        """ Sigmoid Clipping to Signal """
        b = 2.0
        Y = 2*((1 / (1 + np.exp(-b*X))) - 0.5 )
        return Y



    

