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
import FXSetup


        #### MAIN EXECUTABLE ####

if __name__ == "__main__":
    
    ModuleSet = FXSetup.EffectsEmulatorSystem("MySetup")
    ModuleSet.Add(Modules.AnalysisFames("Input",inputShape=(1,44100))

