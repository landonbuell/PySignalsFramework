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
    nSamples = 100000
    sampleRate = 44100
    timeAxis = np.arange(0,nSamples,dtype=np.float32)/sampleRate
    audio = AudioTools.WavesGenerator(time=timeAxis,linearFrequencies=[400,200])
    signalRaw = audio.SineWave()
    #AudioTools.Plotting.PlotTimeSeries(timeAxis,signalRaw,"signal")

    # Create the FX module + Layers
    System = EffectsSystem.EffectsSystem("MySetup")

    FramesLayer = Layers.AnalysisFramesConstructor("ToFrames",inputShape=(nSamples,),
                                                  samplesPerFrame=2048,percentOverlap=0.75,
                                                  maxFrames=512,tailPad=4096,headPad=2048)
    WindowLayer = Layers.WindowFunction("Hanning",nSamples=2048,tailPad=4096,headPad=2048,
                                        function=AudioTools.WindowFunctions.HanningWindow)

    DFTLayer = Layers.DiscreteFourierTransform("DFTLayer",inputShape=(8192,))

    PlotLayer1 = Layers.PlotSignal("PlotFrames",show=True,save=False)
    PlotLayer2 = Layers.PlotSignal("PlotWindow",show=True,save=False)
    PlotLayer3 = Layers.PlotSignal("PlotFrequency",show=True,save=False)
    PlotLayer4 = Layers.PlotSignal("PlotWindow",show=True,save=False)


    # Add Layers to the System
    System.Add(FramesLayer)
    System.Add(PlotLayer1)
    System.Add(WindowLayer)
    System.Add(PlotLayer2)
    System.Add(DFTLayer)
    System.Add(PlotLayer3)

    System.InitializeChain(inputShape=(nSamples,))
    PlotLayer3.SetxAxisData(DFTLayer.GetFreqAxis)

    signalProcessed = System.Call(signalRaw)

    

    print("=)")

