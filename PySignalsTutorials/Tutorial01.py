"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalsTutorialz
File:           Tutorial01.py
Description:
"""

        #### IMPORTS ####

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import PySignalsFramework.AudioTools as AudioTools
import PySignalsFramework.LayersStandard as Layers

if __name__ == "__main__":

   
    """
    Lets use the 'SimpleWavesGenerator' static class from the 'AudioTools' namespace
    Create two simple sinusoidal waves modeled by:
        y_a(t) = cos(2*pi*50*t)
        y_b(t) = 2*sin(2*pi*100*t)  
    """
    
    nSamples = int(2**12)
    timeAxis = np.arange(0,nSamples,1)
    signalA = AudioTools.SimpleWavesGenerator.CosineWave(timeAxis,amp=1,freq=50)
    signalB = AudioTools.SimpleWavesGenerator.SineWave(timeAxis,amp=2,freq=100)
    
    """ 
    To Analyze the signals in frequency space, we apply a Discrete-Fourier-Transform (DFT)
    To each signal to find the frequency components.
    Let's make an Instance of the 'DiscreteFourierTransformLayer' from the 'LayersStandard' namespace
    We give the layer a name and an input shape, and 'initialize' it for usage
    """

    sampleRate = 1024
    layerDFT = Layers.DiscreteFourierTransform("DFT_Layer",inputShape=(nSamples,))
    layerDFT.initialize()

    """ 
    We apply the DFT by passing it through the layer instance with the 'call' Method
    The Array that we pass in (signalA or signalB) will be converted into a Instance of the
        PySignalsFramework.AudioTools.Signal() class and will be given a default Sample Rate of 44100 Hz
    We get the output of the layer by using the 'getSignal()' method
    """

    layerDFT.call(signalA)
    spectrumA = layerDFT.getSignal()

    layerDFT.Call(signalB)
    spectrumB = layerDFT.getSignal()

    """
    To make sure it worked, we can plot the result using the 'PlotSignal' layer from the 'LayersStandard' namespace
    This will let us visualize the resulting Signal in Frequncy Space
    We also need to get the frequency-space axis from the DFT layer to use as the x-axis of the plot layer
    """


    sys.exit(0)