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
    timeAxis = np.arange(0,nSamples)/sampleRate
    audio = AudioTools.SimpleWavesGenerator(time=timeAxis,linearFrequencies=[55,110])
    signalSine = audio.SineWave()
    AudioTools.Plotting.PlotTimeSeries(timeAxis,signalSine,"signal")

    # Create the FX module
    ModuleSet = EffectsSystem.EffectsSystem("MySetup")
    ModuleSet.Add(ModulesTimeSeries.AnalysisFames("InputToFrames",inputShape=(1,nSamples),
                                                  samplesPerFrame=2048,overlapSamples=1024,
                                                  maxFrame=512,zeroPad=2048))

    ModuleSet.InitializeChain()

