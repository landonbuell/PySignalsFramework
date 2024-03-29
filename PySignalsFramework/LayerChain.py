"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           LayerChain.py
Description:
"""

            #### IMPORTS ####

import PySignalsFramework.LayersStandard as Layers
import PySignalsFramework.LayersCustom as Custom

            #### CLASS DEFINITIONS ####

class LayerChainLinear :
    """ 
    Linear Layer Chain Type - 
        Acts as double Linked list of Layers
    --------------------------------
    _name       string          Name to Identify chain instance
    _type       string          Str type of layer chain
    _head       LayerIO         Head Layer in Chain [readonly]
    _tail       LayerIO         Tail Layer in Chain [readonly]
    _size       int             Number of Non-Sentinel nodes in the chain
    --------------------------------
    Return Instantiated LinearLayerChain
    """
    def __init__(self,name,existingLayers=None):
        """ Constructor for LinearLayerChain Instance """
        self._name = name
        self._type = "LayerChainLinear"
        self._head = Layers.IOLayer("HeadNode")
        self._tail = Layers.IOLayer("TailNode")
        self._size = 0
       
        if existingLayers:
            # If given a list of layers
            if type(existingLayers) == list:   
                self.constructFromList(existingLayers)
            elif type(existingLayers) == Layers.AbstractLayer:
                self.constructFromNode(existingLayers)
            else:
                errMsg = "Existing layer must be of type List or AbstractParentLayer, but got {0}".format(type(existingLayers))
                raise TypeError(errMsg)
        else:
            self._head._next = self._tail
            self._tail._prev = self._head

    """ Private Methods """

    def constructFromList(self,layerList):
        """ Assemble a layer chain from list of unconnected layers """
        for layer in layerList:
            self.append(layer)
        return self

    def constructFromNode(self,layerNode):
        """ Assmble a layer chain from single node to end """
        currentLayer = layerNode        
        self.HeadNode.CoupleToNext(layerNode)
        while (currentLayer.Next):
            currentLayer = currentLayer.Next
        self.TailNode.CoupleToPrev(currentLayer)
        return self

    """ Public Interface """

    def append (self,newLayer):
        """ Append a new Layer to the tail of this Chain """
        if (self._size == 0):
            self._head._next = newLayer
            self._tail._prev = newLayer
            newLayer._prev = self.getHead
            newLayer._next = self.getTail
        else:
            oldTail = self.getOutput
            oldTail._next = newLayer
            newLayer._prev = oldTail
            newLayer._next = self._tail
            self._tail._prev = newLayer
        self._size += 1
        return self

    def prepend(self,newLayer):
        """ Prepend a new Layer to the head of this chain """
        if (self._size == 0):
            self._head._next = newLayer
            self._tail._prev = newLayer
            newLayer.Prev = self.getHead
            newLayer.Next = self.getTail
        else:
            oldHead = self.getInput
            oldHead._prev = newLayer
            newLayer._next = oldHead
            newLayer._prev = self._head
            self._head._next = newLayer
        self._size += 1
        return self

    def popFromHead(self):
        """ Remove + Return the first layer after the head """
        if len(self.getChainList) == 0:
            raise IndexError("No elements currently in layer chain")
        else:
            oldInput = self.getInput
            newInput = oldInput.Next
            self.getHead.SetNext(newInput)
            newInput.SetPrev(self.getHead)
            self._size -= 1
        return oldInput

    def popFromTail(self):
        """ Remove + Return the last layer before the tail """
        if len(self.getChainList) == 0:
            raise IndexError("No elements currently in layer chain")
        else:
            oldOutput = self.getOutput
            newOutput = oldOutput.Prev
            self.getTail.SetPrev(newOutput)
            newOutput.SetNext(self.getTail)
            self._size -= 1
        return oldOutput

    def initialize(self,inputShape):
        """ Initialize layer Chain For Usage """
        if (self._size == 0):   # nothing to init
            return self         

        currentIdx = 0
        currentLayer = self.getInput
        while(currentLayer != self.getTail):    # visit each node in the chain
            currentLayer.SetIndex(currentIdx)   # set layer index
            currentLayer.Initialize(inputShape) # init with input Shape

            inputShape = currentLayer.GetOutputShape    # Get new signal shape
            currentLayer = currentLayer.Next            # Layer Layer in the chain
            currentIdx += 1                     # inc index
        # Set Index for Head + Tail Nodes
        self.getHead.SetIndex(-1)
        self.getTail.SetIndex(-1)
        return self
        
    def call(self,X):
        """ Call layer Chain w/ inputs X """
        currentLayer = self.getInput
        while (currentLayer != self.getTail):
            X = currentLayer.Call(X)
            currentLayer = currentLayer.Next
        return X

    def copyChain(self,newName):
        """ Return a Non-aliased copy of this FX Chain """
        return LayerChainLinear(newName,self.getChainList())

    """ Getter & Setter Methods """

    def getInput (self):
        """ Return Input of Layer Chain """
        if self._size == 0:
            return self._head
        else:
            return self._head._next

    def getOutput(self):
        """ Return Output of Layer Chain """
        if self._size == 0:
            return self._tail
        else:
            return self._tail._prev

    def getHead(self):
        """ Return head Layer of Chain """
        return self._head

    def getTail(self):
        """ Return tail Layer of Chain """
        return self._tail

    def getChainList(self):
        """ Return the Lienar Chain Layers as a List """
        chainList = []
        currentLayer = self.getInput
        while (currentLayer != self._tail):
            chainList.append(currentLayer)
            currentLayer = currentLayer._next
        return chainList

    def setSampleRate(self,x):
        """ Set All Layers to have same sample rate """
        currentLayer = self.getHead
        while(currentLayer != None):
            if (type(currentLayer != ResampleLayer)):
                currentLayer.SetSampleRate(x)
            else:
                break
        return self

    """ Magic Methods """

    def __iter__(self):
        """ Iterate through this Layer chain """
        currentLayer = self.getInput
        while (currentLayer != self._tail):
            yield ccurrentLayer
            currentLayer = currentLayer._next
        return self._tail

    def __len__(self):
        """ Get Length of this Layer chain """
        return self._size;

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(len(self)) + " nodes"

