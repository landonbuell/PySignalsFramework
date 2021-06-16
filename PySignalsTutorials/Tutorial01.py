"""
PySignalsTutorials
Tutorial 00 - Basic Effects
Landon Buell
15 June 2021
"""

        #### IMPORTS ####

import os
import sys

import PySignalsFramework as pySig

if __name__ == "__main__":

   
    """
    Import an audio Track from a local source (this repo!)
    Read it in as a numpy array so we can process it
    """
    
    """
    Create Empty PySignals "LinearSystem"
    A "LinearSystem" is a doubly-linked-list of 'layers'
    It has exactly one input, and one output
    """
    linearSystem = pySig.EffectsSystem.LinearSystem("mySystem")


    sys.exit(0)