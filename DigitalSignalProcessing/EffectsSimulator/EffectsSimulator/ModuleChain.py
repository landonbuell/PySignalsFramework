"""
Landon Buell
EffectsEmmulatorPython
ModuleChain
5 Feb 2020
"""

            #### IMPORTS ####

import ModulesTimeSeries
import ModulesFrequencySeries

            #### CLASS DEFINITIONS ####

class LinearModuleChain :
    """ 
    Linear Module Chain Type - Acts as double inked list
    --------------------------------
    _name (str) : 
    _type (str) : Type of Module Chain

    _head (ModuleAbstract) : Head Module in Chain [readonly]
    _tail (ModuleAbstract) : Tail Module in Chain [readonly]

    --------------------------------
    Return Instantiated LinearModuleChain
    """
    def __init__(self,name,existingModules=None):
        """ Constructor for LinearModuleChain Instance """
        self._head = Modules.IdentityModule("HeadNode")
        self._tail = Modules.IdentityModule("TailNode")
        if existingModules:
            for mod in existingModules:
                self.Append(mod)
        else:
            self._head._next = self._tail
            self._tail._prev = self._head

    def Append (self,newModule):
        """ Append a new Module to the tail of this Chain """
        oldTail = self._tail._prev
        oldTail._next = newModule
        newModule._prev = oldTail
        newModule._next = self._tail
        self._tail._prev = newModule
        return self

    def Prepend(self,newModule):
        """ Prepend a new Module to the head of this chain """
        oldHead = self._head._next
        oldHead._prev = newModule
        newModule._next = oldHead
        newModule._prev = self._head
        self._head._next = newModule
        return self

    def Call(self,X):
        """ Call module Chain w/ inputs X """
        currentModule = self._head._next
        while (currentModule != self._tail):
            X = currentModule.Call(X)
            currentModule = currentModule._next
        return X

    @property
    def GetInput (self):
        """ Return Input of Module Chain """
        return self._head._next

    @property
    def GetOutput(self):
        """ Return Output of Module Chain """
        return self._tail._prev

    @property
    def GetChainList(self):
        """ Return the Lienar Chain Modules as a List """
        chainList = []
        currentModule = self._head._next
        while (currentModule._next != self._tail):
            chainList.append(currentModule)
            currentModule = currentModule._next
        return chainList

    def CopyChain(self,newName):
        """ Return a Non-aliased copy of this FX Chain """
        return LinearModuleChain(newName,self.GetChainList)

    def __len__(self):
        """ Get Length of this Module chain """
        return len(self.GetChainList)

    

class ModuleChainTools :
    """
    Class of Tools to traverse and Interact with a Modules or Chain 
    """

    @staticmethod
    def TraverseModuleChain(currentModule):
        """ Traverse a chain of modules, return tail node """
        while currentModule._next != None:
            currentModule = currentModule._next
        return currentModule

