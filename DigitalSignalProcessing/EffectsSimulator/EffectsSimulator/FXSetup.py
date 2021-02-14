"""
Landon Buell
EffectsEmmulatorPython
FXSetup
7 Feb 2020
"""

        #### IMPORTS ####

import ModuleChain
import Modules

        #### CLASS DEFINITIONS ####

class EffectsEmulatorSystem:
    """ 
    EffectsEmulatorSystem Type -
        Parent container for all classes in this package
    --------------------------------
    _name (str) : Name for user- Identification
    _type (str) : Type of EffectsEmulatorSystem
    
    _moduleChain (LinearModuleChain) : 
    _nModules (int) : Number of modules in chain
    
    --------------------------------
    Return Instatiated EffectsEmmulatorSystem
    """

    def __init__(self,name,modules=None,):
        """ Constructor for EffectsEummulatorSystem """
        self._name = name
        self._type = "EffectsEmmulatorSystem"

        self._moduleChain = FXChain.LinearModuleChain(name+"_chain")

    

    def Call(self,X):
        """ Call Each layer in chain with Inputs X """
        return X

    def InitializeChain(self):
        """ Initialize All modules in the Module Chain """
        currentModule = self._moduleChain._head._next
        while (currentModule != self._moduleChain._tail):
            currentModule.Initialize()
            currentModule = currentModule._next
        return self

    def Add(self,newModule):
        """ Add a new Module to this Module Chain """
        self._moduleChain.Append(newModule)

        return self

    @property
    def Input(self):
        """ Get input module in Chain """
        return self._moduleChain.GetInput

    @property
    def Output(self):
        """ Get last module in chain """

    def GetInputParams(self,inputSignal):
        """ Detemrine qualities of input signal """
        return self

    def SetInputParams(self,paramsList):
        """ Set Parameters of each layer """
        return self


