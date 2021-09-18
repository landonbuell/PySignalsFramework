"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           EffectsSystem.py
Description:
"""

        #### IMPORTS ####

import PySignalsFramework.AdminTools as AdminTools

        #### CLASS DEFINITIONS ####

class LinearSystem:
    """ 
    EffectsSystem Type -
        Parent container for all classes in this package
        Contains One input and One Output Node
    --------------------------------
    _name           str                 Name for user- Identification
    _type           str                 Type of EffectsSystem
    _isInit         bool                T/F if layer chain is initialized for use
    _layerChain     DoubleLinkedList    Double Linked List Layer Chain
    --------------------------------
    Return Instatiated EffectsSystem
    """
    
    def __init__(self,name,layers=None,input=None,output=None):
        """ Constructor for EffectsEummulatorSystem """
        self._name          = name
        self._type          = "EffectsSystem"
        self._layerChain    = AdminTools.DoubleLinkedList()
        self._isInit        = False

    """ Public Interface """

    def addTail(self,newLayer):
        """ Add a new Layer to the end of the Layer Chain """
        self._layerChain.append(newLayer)
        self._isInit = False
        return self

    def addHead(self,newLayer):
        """ Add a new Layer to the front of the Layer Chain """
        self._layerChain.prepend(newLayer)
        self._isInit = False
        return self

    def popTail(self):
        """ Remove a layer from the end of the layer chain """
        removedLayer =  self._layerChain.popTail()
        self._isInit = False
        return removedLayer

    def popHead(self):
        """ Remove a layer from the front of the layer chain """
        removedLayer =  self._layerChain.popHead()
        self._isInit = False
        return removedLayer

    def initializeChain(self,inputShape):
        """ Initialize All modules in the Layer Chain """
        successful = False
        for layer in self._layerChain:  # iterate through each layer
            layer.initialze()
        successful = True
        return successful

    def call(self,X):
        """ Call Each layer in chain with Inputs X """
        if (self._isInit != True):
            raise Exception("Chain Not Initialized!")
        if (X.shape != self.getInputShape):     # shapes are not equal
            self.initializeChain(X.shape)       # Re-init Chain w/ shape

        # Call the Layer Chain
        for layer in self._layerChain:
            layer.call(X)           # call w/ inputs X
            X = layer.getSignal()   # get resulting signal

        # The Final Signal
        return X
       
    """ Getter & Setter Methods """

    def getName(self):
        """ Get the Name of this Effects System """
        return self._name

    def getType(self):
        """ Get Type of this Effects System """
        return self._type

    def getInput(self):
        """ Get input Layer of Chain """
        return self._layerChain.getInput()

    def getOutput(self):
        """ Get output Layer of Chain """
        return self._layerChain.GetOutput()

    def getInputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.getInput().GetInputShape()

    def getOutputShape(self):
        """ Return the inputShape of this System """
        return self._layerChain.getOutput().getOutputShape()

    """ Magic Methods """

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(self.ChainSize) + " node(s)"

    def __len__(self):
        """ Return Number of layers in the module """
        return self._nLayers
