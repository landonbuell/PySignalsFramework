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
    nSamples = len(signalRaw)

    # Create the FX module + Layers
    System = EffectsSystem.EffectsSystem("MySetup")

    NormalizeLayer = Layers.NormalizeLayer("NormLayer")
    OverdriveLayer = Custom.SigmoidOverdrive("SigmoidDrive")

    PlotLayer1 = Layers.PlotSignal("PlotRawSignal",show=True,save=False)
    PlotLayer2 = Layers.PlotSignal("PlotNewSignal",show=True,save=False)

    # Add Layers to the System
    System.Add(NormalizeLayer)
    System.Add(PlotLayer1)
    System.Add(OverdriveLayer)
    System.Add(PlotLayer2)

    System.InitializeChain(inputShape=(nSamples,))

    signalProcessed = System.Call(signalNorm)

    AudioTools.AudioIO.WriteWAV(outputAudioFile,signalProcessed,sampleRate)

    print("=)")


