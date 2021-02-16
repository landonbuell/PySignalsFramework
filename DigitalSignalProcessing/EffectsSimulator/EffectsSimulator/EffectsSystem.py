"""
Landon Buell
EffectsEmmulatorPython
FXSetup
7 Feb 2020
"""

        #### IMPORTS ####

import ModuleChain

        #### CLASS DEFINITIONS ####

class EffectsSystem:
    """ 
    EffectsSystem Type -
        Parent container for all classes in this package
    --------------------------------
    _name (str) : Name for user- Identification
    _type (str) : Type of EffectsSystem
    
    _moduleChain (LinearModuleChain) : 
    _nModules (int) : Number of modules in chain
    
    --------------------------------
    Return Instatiated EffectsSystem
    """

    def __init__(self,name,modules=None,):
        """ Constructor for EffectsEummulatorSystem """
        self._name = name
        self._type = "EffectsEmmulatorSystem"

        self._moduleChain = ModuleChain.LinearModuleChain(name+"_chain")

    def Call(self,X):
        """ Call Each layer in chain with Inputs X """
        currentModule = self.Input
        while (currentModule != self._moduleChain.GetTail):
            X = currentModule.Call(X)
            currentModule = currentModule._next
        return X

    def InitializeChain(self,inputShape=None):
        """ Initialize All modules in the Module Chain """
        currentIndex = 1
        currentModule = self.Input
        self._moduleChain._head._chainIndex = 0
        if inputShape:
            currentModule.SetInputShape(inputShape)
        while (currentModule != self._moduleChain.GetTail):
            currentModule.Initialize()
            currentModule._chainIndex = currentIndex
            currentIndex += 1
            currentModule = currentModule._next
        self._moduleChain._tail._chainIndex = currentIndex
        return self

    def Add(self,newModule):
        """ Add a new Module to this Module Chain """
        self._moduleChain.Append(newModule)

        return self

    @property
    def GetChainList(self):
        """ Return the EffectsSystem chain as list of modules """
        return self._moduleChain.GetChainlist

    @property
    def ChainSize(self):
        """ Return the number of elements in the chain list """
        return len(self.__moduleChain)

    @property
    def Input(self):
        """ Get input module in Chain """
        return self._moduleChain.GetInput

    @property
    def Output(self):
        """ Get last module in chain """
        return self._moduleChain.GetOutput

    def GetInputParams(self,inputSignal):
        """ Detemrine qualities of input signal """
        return self

    def SetInputParams(self,paramsList):
        """ Set Parameters of each layer """
        return self

    def PrintSummary(self):
        """ Print a summary of this module chain to Console """
        return self

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(self.ChainSize) + " nodes"
