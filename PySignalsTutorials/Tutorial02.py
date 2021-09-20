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

if __name__ == "__main__":
    

    """
    Lets use the 'SimpleWavesGenerator' static class from the 'AudioTools' namespace again
    Create a signal composed of 4 sinusoids:
        y_a(t) = cos(2*pi*50*t)
        y_b(t) = 2*sin(2*pi*100*t)  
    """
    
    nSamples = 100000
    sampleRate = 44100
    timeAxis = np.arange(0,nSamples,1) / sampleRate

    signalA = AudioTools.SimpleWavesGenerator.CosineWave(timeAxis,amp=2**16,freq=110)
    signalB = AudioTools.SimpleWavesGenerator.CosineWave(timeAxis,amp=2**18,freq=220)
    signalC = AudioTools.SimpleWavesGenerator.CosineWave(timeAxis,amp=2**14,freq=440)
    signalD = AudioTools.SimpleWavesGenerator.CosineWave(timeAxis,amp=2**14,freq=8800)
    
    result01 = signalA + signalB + signalC + signalD
    result01 = result01.astype(np.int32)

    """
    Plot the Signal in Time-space so that we are sure that we have it to our liking
    Use the "plotTimeSeries" method from the "Plotting" Static Class
    Also, print the data type so we know how to format it when we read the audio back in
    """

    AudioTools.Plotting.plotTimeSeries(timeAxis,result01,title="Tutorial02 - Result Signal 00")
    print("DataType:" , result01.dtype)

    """
    Finally, we export the signal array or numpy object
    Note that when Exported, the signal will be cast to type np.float32
    """

    exportPath01 = "../AudioSamples/signalResult01.wav"
    AudioTools.AudioIO.writeWAV(exportPath01,result01,sampleRate)
    print("DataType:" , result01.dtype)