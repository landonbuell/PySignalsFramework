"""
Landon Buell
EffectsSimulator
LayerChain
5 Feb 2020
"""

            #### IMPORTS ####

import LayersStandard as Layers
import LayersCustom as Custom

            #### CLASS DEFINITIONS ####

class LayerChainLinear :
    """ 
    Linear Layer Chain Type - 
        Acts as double Linked list of Layers
    --------------------------------
    _name (str) : 
    _type (str) : Type of Layer Chain

    _head (LayerAbstract) : Head Layer in Chain [readonly]
    _tail (LayerAbstract) : Tail Layer in Chain [readonly]

    _size (int) : Number of non-sentinel nodes in Layer
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
                self.AssembleFromList(existingLayers)
            elif type(existingLayers) == Layers.AbstractLayer:
                self.AssembleFromNode(existingLayers)
            else:
                errMsg = "Existing layer must be of type List or AbstractParentLayer, but got {0}".format(type(existingLayers))
                raise TypeError(errMsg)
        else:
            self._head._next = self._tail
            self._tail._prev = self._head

    """ Private Methods """

    def AssembleFromList(self,layerList):
        """ Assemble a layer chain from list of unconnected layers """
        for layer in layerList:
            self.Append(layer)
        return self

    def AssembleFromNode(self,layerNode):
        """ Assmble a layer chain from single node to end """
        currentLayer = layerNode        
        self.HeadNode.CoupleToNext(layerNode)
        while (currentLayer.Next):
            currentLayer = currentLayer.Next
        self.TailNode.CoupleToPrev(currentLayer)
        return self

    """ Public Interface """

    def Append (self,newLayer):
        """ Append a new Layer to the tail of this Chain """
        if (self._size == 0):
            self._head._next = newLayer
            self._tail._prev = newLayer
            newLayer._prev = self.GetHead
            newLayer._next = self.GetTail
        else:
            oldTail = self.GetOutput
            oldTail._next = newLayer
            newLayer._prev = oldTail
            newLayer._next = self._tail
            self._tail._prev = newLayer
        self._size += 1
        return self

    def Prepend(self,newLayer):
        """ Prepend a new Layer to the head of this chain """
        if (self._size == 0):
            self._head._next = newLayer
            self._tail._prev = newLayer
            newLayer.Prev = self.GetHead
            newLayer.Next = self.GetTail
        else:
            oldHead = self.GetInput
            oldHead._prev = newLayer
            newLayer._next = oldHead
            newLayer._prev = self._head
            self._head._next = newLayer
        self._size += 1
        return self

    def PopFromTail(self):
        """ Remove + Return the last layer before the tail """
        if len(self.GetChainList) == 0:
            raise IndexError("No elements currently in layer chain")
        else:
            oldOutput = self.GetOutput
            newOutput = oldOutput.Prev
            self.GetTail.SetPrev(newOutput)
            newOutput.SetNext(self.GetTail)
            self._size -= 1
        return oldOutput

    def PopFromHead(self):
        """ Remove + Return the first layer after the head """
        if len(self.GetChainList) == 0:
            raise IndexError("No elements currently in layer chain")
        else:
            oldInput = self.GetInput
            newInput = oldInput.Next
            self.GetHead.SetNext(newInput)
            newInput.SetPrev(self.GetHead)
            self._size -= 1
        return oldInput

    def Initialize(self,inputShape):
        """ Initialize layer Chain For Usage """
        if (self._size == 0):   # nothing to init
            return self         

        currentIdx = 0
        currentLayer = self.GetInput
        while(currentLayer != self.GetTail):    # visit each node in the chain
            currentLayer.SetIndex(currentIdx)   # set layer index
            currentLayer.Initialize(inputShape) # init with input Shape

            inputShape = currentLayer.GetOutputShape    # Get new signal shape
            currentLayer = currentLayer.Next            # Layer Layer in the chain
            currentIdx += 1                     # inc index
        # Set Index for Head + Tail Nodes
        self.GetHead.SetIndex(-1)
        self.GetTail.SetIndex(-1)
        return self
        
    def Call(self,X):
        """ Call layer Chain w/ inputs X """
        currentLayer = self.GetInput
        while (currentLayer != self.GetTail):
            X = currentLayer.Call(X)
            currentLayer = currentLayer.Next
        return X

    def CopyChain(self,newName):
        """ Return a Non-aliased copy of this FX Chain """
        return LayerChainLinear(newName,self.GetChainList())

    """ Getter & Setter Methods """

    @property
    def GetInput (self):
        """ Return Input of Layer Chain """
        if self._size == 0:
            return self._head
        else:
            return self._head._next

    @property
    def GetOutput(self):
        """ Return Output of Layer Chain """
        if self._size == 0:
            return self._tail
        else:
            return self._tail._prev

    @property
    def GetHead(self):
        """ Return head Layer of Chain """
        return self._head

    @property
    def GetTail(self):
        """ Return tail Layer of Chain """
        return self._tail

    def GetChainList(self):
        """ Return the Lienar Chain Layers as a List """
        chainList = []
        currentLayer = self.GetInput
        while (currentLayer != self._tail):
            chainList.append(currentLayer)
            currentLayer = currentLayer._next
        return chainList

    def SetSampleRate(self,x):
        """ Set All Layers to have same sample rate """
        currentLayer = self.GetHead
        while(currentLayer != None):
            if (type(currentLayer != ResampleLayer)):
                currentLayer.SetSampleRate(x)
            else:
                break
        return self

    """ Magic Methods """

    def __iter__(self):
        """ Iterate through this Layer chain """
        currentLayer = self.GetInput
        while (currentLayer != self._tail):
            yield ccurrentLayer
            currentLayer = currentLayer._next

    def __len__(self):
        """ Get Length of this Layer chain """
        return self._size;

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(len(self)) + " nodes"

