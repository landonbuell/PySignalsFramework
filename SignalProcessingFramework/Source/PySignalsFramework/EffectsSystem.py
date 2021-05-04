"""
Landon Buell
EffectsEmmulatorPython
FXSetup
7 Feb 2020
"""

        #### IMPORTS ####

import PySignalsFramework.LayerChain as LayerChain

        #### CLASS DEFINITIONS ####

class EffectsSystem:
    """ 
    EffectsSystem Type -
        Parent container for all classes in this package
    --------------------------------
    _name (str) : Name for user- Identification
    _type (str) : Type of EffectsSystem
    
    _layerChain (LayerChainLinear) : 
    _nLayers (int) : Number of layers in chain

    _isInit (bool) : T/F if layer chain is initialized for use
    --------------------------------
    Return Instatiated EffectsSystem
    """
    
    def __init__(self,name,layers=None,input=None,output=None):
        """ Constructor for EffectsEummulatorSystem """
        self._name = name
        self._type = "EffectsSystem"

        self._layerChain = LayerChain.LayerChainLinear(name+"Chain",layers)
        self._nLayers = len(self._layerChain)

        self._isInit = False

    def InitializeChain(self,inputShape):
        """ Initialize All modules in the Layer Chain """
        self._layerChain.Initialize(inputShape)
        self._isInit = True
        return self

    def Add(self,newLayer):
        """ Add a new Layer to this Layer Chain """
        self._layerChain.Append(newLayer)
        self._nLayers += 1
        self._isInit = False
        return self

    def Pop(self):
        """ Remove a layer from the end of the layer chain """
        removedLayer =  self._layerChain.PopFromTail()
        self._nLayers -= 1;
        self._isInit = False
        return removedLayer

    def Call(self,X):
        """ Call Each layer in chain with Inputs X """
        if (self._isInit != True):
            raise Exception("Chain No Initialized!")
        if (X.shape != self.GetInputShape):     # shapes are not equal
            self.InitializeChain(X.shape)       # Re-init Chain w/ shape

        currentLayer = self.Input
        while (currentLayer != self._layerChain.GetTail):
            X = currentLayer.Call(X)
            currentLayer = currentLayer.Next
        return X

    def GetChainList(self):
        """ Return the EffectsSystem chain as list of modules """
        return self._layerChain.GetChainlist()

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

    @property
    def GetInputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.GetInput.GetInputShape

    @property
    def GetOutputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.GetOuput.GetOutputShape

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
        return self._type + ": \'" + self._name + "\' w/ " + str(self.ChainSize) + " node(s)"
