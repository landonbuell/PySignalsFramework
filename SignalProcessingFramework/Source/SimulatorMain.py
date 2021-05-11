"""
Landon Buell
EffectsEmmulatorPython
Main Executable
5 Feb 2020
"""

        #### IMPORTS ####

import os
import sys
import numpy as np

import PySignalsFramework.Layers as Layers
import PySignalsFramework.EffectsSystem as EffectsSystem
import PySignalsFramework.AudioTools as AudioTools

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    # Load in sample signal
    nSamples = 10000
    sampleRate = 1000
    timeAxis = np.arange(0,nSamples,dtype=np.float32)/sampleRate
    audio = AudioTools.WavesGenerator(time=timeAxis,linearFrequencies=[10,20])
    signalRaw = audio.SineWave()
    #AudioTools.Plotting.PlotTimeSeries(timeAxis,signalRaw,"signal")

    # Create the FX module + Layers
    System = EffectsSystem.EffectsSystem("MySetup")

    FramesLayer = Layers.AnalysisFramesConstructor("ToFrames",inputShape=(1,nSamples),
                                                  samplesPerFrame=1024,percentOverlap=0.75,
                                                  maxFrames=512,tailPad=2048,headPad=1024)
    WindowLayer = Layers.WindowFunction("Hanning",nSamples=1024,tailPad=2048,headPad=1024,
                                        function=AudioTools.WindowFunctions.HanningWindow)

    DFTLayer = Layers.DiscreteFourierTransform("DFTLayer",)

    PlotLayer1 = Layers.PlotSignal("PlotFrames",show=True)
    PlotLayer2 = Layers.PlotSignal("PlotWindow",show=True)
    PlotLayer3 = Layers.PlotSignal("PlotFrequency",show=True)
    PlotLayer4 = Layers.PlotSignal("PlotWindow",show=True)


    # Add Layers to the System
    System.Add(FramesLayer)
    System.Add(PlotLayer1)
    System.Add(WindowLayer)
    System.Add(PlotLayer2)
    System.Add(DFTLayer)

    System.InitializeChain(inputShape=(nSamples,))

    signalProcessed = System.Call(signalRaw)

    

    print("=)")

