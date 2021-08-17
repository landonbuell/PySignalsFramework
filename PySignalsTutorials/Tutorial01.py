"""
PySignalsTutorials
Tutorial 00 - 
Landon Buell
15 June 2021
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
    We give the layer a name, sample rate, input shape, and 'initialize' it for usage
    """

    sampleRate = 1024
    layerDFT = Layers.DiscreteFourierTransform("DFT_Layer",sampleRate=sampleRate,inputShape=(nSamples,))
    layerDFT.initialize(inputShape=(nSamples,))

    """ 
    We apply the DFT by passing it through the layer instance with the 'call' Method
    We get the output of the layer by using the 'getSignal()' method
    """

    layerDFT.Call(signalA)
    spectrumA = layerDFT.getSignal()

    layerDFT.Call(signalB)
    spectrumB = layerDFT.getSignal()

    plt.plot(layerDFT.getFreqAxis(),spectrumA)
    plt.show()

    """
    To make sure it worked, we can plot the result using the 'PlotSignal' layer from the 'LayersStandard' namespace
    This will let us visualize the resulting Signal to visualize.
    We also need to get the frequency-space axis from the DFT layer to use as the x-axis of the plot layer
    """

    freqAxis = layerDFT.getFreqAxis()
    layerPlotSpectrum = Layers.PlotSignal("Plot_Layer",sampleRate=sampleRate,inputShape=(nSamples,),
                                  show=True,save=False,xAxis=freqAxis)
    layerPlotSpectrum.initialize(inputShape=(nSamples,))

    layerPlotSpectrum.call(spectrumA)
    layerPlotSpectrum.call(spectrumB)

    sys.exit(0)