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

    """ Public Interface """

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

    def InitializeChain(self,inputShape):
        """ Initialize All modules in the Layer Chain """
        self._layerChain.Initialize(inputShape)
        self._isInit = True
        return self

    def Call(self,X):
        """ Call Each layer in chain with Inputs X """
        if (self._isInit != True):
            raise Exception("Chain Not Initialized!")
        if (X.shape != self.GetInputShape):     # shapes are not equal
            self.InitializeChain(X.shape)       # Re-init Chain w/ shape

        # Call the Layer Chain
        X = self._layerChain.Call(X)
        return X
       
    """ Getter & Setter Methods """

    @property
    def Name(self):
        """ Get the Name of this Effects System """
        return self._name

    @property
    def Type(self):
        """ Get Type of this Effects System """
        return self._type

    def GetLayerChainInst(self):
        """ Get the LayerChain as the Instance """
        return self._layerChain

    def GetLayerChainList(self):
        """ Get the LayerChain as a List """
        return self._layerChain.GetChainlist()

    @property
    def ChainSize(self):
        """ Return the number of elements in the chain list """
        return len(self._layerChain)

    @property
    def Input(self):
        """ Get input Layer of Chain """
        return self._layerChain.GetInput

    @property
    def Output(self):
        """ Get output Layer of Chain """
        return self._layerChain.GetOutput

    @property
    def GetInputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.GetInput.GetInputShape

    @property
    def GetOutputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.GetOuput.GetOutputShape

    def SetSampleRate(self,x):
        """ Set All Layers to new Sample Rate """
        self._layerChain.SetSampleRate(x)
        return self

    """ Magic Methods """

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(self.ChainSize) + " node(s)"
