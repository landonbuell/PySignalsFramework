"""
Landon Buell
EffectsSimulator
LayerChain
5 Feb 2020
"""

            #### IMPORTS ####

from LayersGeneric import *

            #### CLASS DEFINITIONS ####

class LayerChainLinear :
    """ 
    Linear Layer Chain Type - Acts as double inked list
    --------------------------------
    _name (str) : 
    _type (str) : Type of Layer Chain

    _head (LayerAbstract) : Head Layer in Chain [readonly]
    _tail (LayerAbstract) : Tail Layer in Chain [readonly]

    --------------------------------
    Return Instantiated LinearLayerChain
    """
    def __init__(self,name,existingLayers=None):
        """ Constructor for LinearLayerChain Instance """
        self._name = name
        self._type = "LayerChainLinear"
        self._head = IdentityLayer("HeadNode")
        self._tail = IdentityLayer("TailNode")
       
        if existingLayers:
            # If given a list of layers
            if type(existingLayers) == list:   
                self.AssembleFromList(existingLayers)
            elif type(existingLayers) == AbstractParentLayer:
                self.AssembleFromNode(existingLayers)
            else:
                errMsg = "Existing layer must be of type List or AbstractParentLayer, but got {0}".format(type(existingLayers))
                raise TypeError(errMsg)
        else:
            self._head._next = self._tail
            self._tail._prev = self._head

    def AssembleFromList(self,layerList):
        """ Assemble a layer chain from list of unconnected layers """
        for layer in layerList:
            self.Append(layer)
        return self

    def AssembleFromNode(self,layerNode):
        """ Assmble a layer chain from single node to end """
        currentLayer = layerNode        
        while True:
            self.Append(currentLayer)
            if currentLayer._next is None: 
                break
            currentLayer = currentLayer._next
        return self


    def Append (self,newLayer):
        """ Append a new Layer to the tail of this Chain """
        oldTail = self._tail._prev
        oldTail._next = newLayer
        newLayer._prev = oldTail
        newLayer._next = self._tail
        self._tail._prev = newLayer
        return self

    def Prepend(self,newLayer):
        """ Prepend a new Layer to the head of this chain """
        oldHead = self._head._next
        oldHead._prev = newLayer
        newLayer._next = oldHead
        newLayer._prev = self._head
        self._head._next = newLayer
        return self

    def Call(self,X):
        """ Call layer Chain w/ inputs X """
        currentLayer = self._head._next
        while (currentLayer != self._tail):
            X = currentLayer.Call(X)
            currentLayer = currentLayer._next
        return X

    @property
    def GetInput (self):
        """ Return Input of Layer Chain """
        return self._head._next

    @property
    def GetOutput(self):
        """ Return Output of Layer Chain """
        return self._tail._prev

    @property
    def GetHead(self):
        """ Return head Layer of Chain """
        return self._head

    @property
    def GetTail(self):
        """ Return tail Layer of Chain """
        return self._tail

    @property
    def GetChainList(self):
        """ Return the Lienar Chain Layers as a List """
        chainList = []
        currentLayer = self._head._next
        while (currentLayer._next != self._tail):
            chainList.append(currentLayer)
            currentLayer = currentLayer._next
        return chainList

    def CopyChain(self,newName):
        """ Return a Non-aliased copy of this FX Chain """
        return LinearLayerChain(newName,self.GetChainList)

    # Magic Methods

    def __len__(self):
        """ Get Length of this Layer chain """
        return len(self.GetChainList)

    def __str__(self):
        """ string-level representation of this instance """
        return self._type + " - " + self._name

    def __repr__(self):
        """ Programmer-level representation of this instance """
        return self._type + ": \'" + self._name + "\' w/ " + str(len(self)) + " nodes"

class LayerChainTools :
    """
    Class of Tools to traverse and Interact with a Layers or Chain 
    """

    @staticmethod
    def TraverseLayerChain(currentLayer):
        """ Traverse a chain of layers, return tail node """
        while currentLayer._next != None:
            currentLayer = currentLayer._next
        return currentLayer

