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

import ModulesTimeSeries
import ModulesFrequencySeries
import EffectsSystem
import AudioTools


        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    # Load in sample signal
    nSamples = 88200
    sampleRate = 44100
    timeAxis = np.arange(0,nSamples,dtype=np.float32)/sampleRate
    audio = AudioTools.SimpleWavesGenerator(time=timeAxis,linearFrequencies=[55,110])
    signalRaw = audio.SineWave()
    #AudioTools.Plotting.PlotTimeSeries(timeAxis,signalRaw,"signal")

    # Create the FX module
    ModuleSet = EffectsSystem.EffectsSystem("MySetup")
    ModuleSet.Add(ModulesTimeSeries.AnalysisFamesConstructor("InputToFrames",inputShape=(1,nSamples),
                                                  samplesPerFrame=2048,percentOverlap=0.75,
                                                  maxFrames=512,zeroPad=2048))

    ModuleSet.InitializeChain()

    signalProcessed = ModuleSet.Call(signalRaw)

    

    print("=)")

