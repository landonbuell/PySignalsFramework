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

import PySignalsFramework.LayersStandard as Layers
import PySignalsFramework.LayersCustom as Custom
import PySignalsFramework.EffectsSystem as EffectsSystem
import PySignalsFramework.AudioTools as AudioTools

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    # Load in sample signal
    inputAudioFile = "C:\\Users\\lando\\Documents\\GitHub\\Signal-Processing-Simulator\\KnockingOnHeavensDoor.wav"
    outputAudioFile = "C:\\Users\\lando\\Documents\\GitHub\\Signal-Processing-Simulator\\KnockingOnHeavensDoorOutput.wav"

    sampleRate,signalRaw = AudioTools.AudioIO.ReadWAV(inputAudioFile)
    nSamples = signalRaw.shape[-1]

    freqSpace = np.fft.fftfreq(4096,1/sampleRate)

    # Create the FX module + Layers
    System = EffectsSystem.EffectsSystem("MySetup")
    
    Clipping = Custom.ClipOverdriveLayer("Clipper",threshold=0.1,softClip=True)
    ToFrames = Layers.AnalysisFramesConstructor("ToFrames",samplesPerFrame=1024,percentOverlap=0.75,
                                                maxFrames=256,tailPad=2048,headPad=1024)
    ToSignal = Layers.AnalysisFramesDestructor("ToSignal",deconstructParams=ToFrames.GetFrameParams)
    ToFreq = Layers.DiscreteFourierTransform("DFT")

    PlotLayer1 = Layers.PlotSignal("PlotRawSignal",show=True,save=False)
    PlotLayer2 = Layers.PlotSignal("PlotNewSignal",show=True,save=False)
    PlotLayer3 = Layers.PlotSignal("PlotFrequency",show=True,save=False)

    # Add Layers to the System
    #System.Add(PlotLayer1)
    System.Add(Clipping)
    #System.Add(PlotLayer2)
    System.Add(ToFrames)
    System.Add(ToFreq)
    System.Add(PlotLayer3)

    System.InitializeChain(inputShape=(nSamples,))

    PlotLayer3.SetxAxisData(freqSpace)
    signalProcessed = System.Call(signalRaw)

    """
    AudioTools.Plotting.PlotTimeSeries( xData = np.arange(nSamples),
                                        yData = np.array([signalRaw,signalProcessed]).transpose(),
                                        labels = ["Raw","Processed"] )

    AudioTools.AudioIO.WriteWAV(outputAudioFile,signalProcessed,sampleRate)
    """

    print("=)")


