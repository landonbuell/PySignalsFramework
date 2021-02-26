"""
Landon Buell
EffectsEmmulatorPython
FXSetup
7 Feb 2020
"""

        #### IMPORTS ####

import LayerChain

        #### CLASS DEFINITIONS ####

class EffectsSystem:
    """ 
    EffectsSystem Type -
        Parent container for all classes in this package
    --------------------------------
    _name (str) : Name for user- Identification
    _type (str) : Type of EffectsSystem
    
    _layerChain (LayerChainLinear) : 
    _nLayers (int) : Number of modules in chain
    
    --------------------------------
    Return Instatiated EffectsSystem
    """

    def __init__(self,name,modules=None,):
        """ Constructor for EffectsEummulatorSystem """
        self._name = name
        self._type = "EffectsSystem"

        self._layerChain = LayerChain.LayerChainLinear(name+"Chain")

    def Call(self,X):
        """ Call Each layer in chain with Inputs X """
        currentLayer = self.Input
        while (currentLayer != self._layerChain.GetTail):
            X = currentLayer.Call(X)
            currentLayer = currentLayer._next
        return X

    def InitializeChain(self,inputShape=None):
        """ Initialize All modules in the Layer Chain """
        currentIndex = 1
        currentLayer = self.Input
        self._layerChain._head._chainIndex = 0
        if inputShape:
            currentLayer.SetInputShape(inputShape)
        while (currentLayer != self._layerChain.GetTail):
            currentLayer.Initialize()
            currentLayer._chainIndex = currentIndex
            currentIndex += 1
            currentLayer = currentLayer._next
        self._layerChain._tail._chainIndex = currentIndex
        return self

    def Add(self,newLayer):
        """ Add a new Layer to this Layer Chain """
        self._layerChain.Append(newLayer)

        return self

    @property
    def GetChainList(self):
        """ Return the EffectsSystem chain as list of modules """
        return self._layerChain.GetChainlist

    @property
    def ChainSize(self):
        """ Return the number of elements in the chain list """
        return len(self._layerChain)

    @property
    def Input(self):
        """ Get input module in Chain """
        return self._layerChain.GetInput

    @property
    def Output(self):
        """ Get last module in chain """
        return self._layerChain.GetOutput

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
