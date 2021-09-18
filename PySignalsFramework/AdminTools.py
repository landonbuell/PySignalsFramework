"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           AdminTools.py
Description:
"""

from __init__ import *
import PySignalsFramework.LayersStandard as Layers

        #### FUNCTION DEFINTIONS ####

def setModule(module):
    """
    Decorator for overriding __module__ on a function or class.
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


        #### ERROR TYPES ####

        #### CLASS DEFINTIONS ####

class DoubleLinkedList:
    """ 
    DoubleLinkedList Type
        Simple Implementation of a double Linked List for collection of layers
    --------------------------------
    _head       LayerIO         Head Layer in Chain [readonly]
    _tail       LayerIO         Tail Layer in Chain [readonly]
    _size       int             Number of Non-Sentinel nodes in the chain
    --------------------------------
    
    """

    def __init__(self,dType=Layers.AbstractLayer,items=None):
        """ Constructor for DoubleLinkedList Instance """
        self._dType     = dType
        self._head      = ListNode(data="HEADNODE")
        self._tail      = ListNode(data="TAILNODE") 
        self._size      = 0

        if (items is not None):  # Given Items
            pass

        else:                   # Empty List couple head<->tail
            DoubleLinkedList.coupleNodes(
                self._head,self._tail)
        
    def __del__(self):
        """ Destructor for DoubleLinkedList Instance """
        pass

    """ Private Interface """

    class ListNode:
        """ Represents Node in Double Linked List """

        def __init__(self,data,prev=None,next=None,):
            """ Constructor for ListNode Instance """
            self._data = data
            self._prev = prev
            self._next = next

        def __del__(self):
            """ Destructor for ListNode Instance """
            pass

        """ Public Interface """

        def getData(self):
            """ Get the Data stored at this ListNode """
            return self._data

        def setData(self,x):
            """ Set the Data stored at this ListNode """       
            self._data = x
            return self

        def getPrev(self):
            """ Get the Previous  Node """
            return self._prev

        def setPrev(self,x):
            """ Set the Previous Node """
            if (type(x) == ListNode):
                self._prev = x
            else:
                raise TypeError("Must be of type 'ListNode'")
            return self

        def getNext(self):
            """ Get the Next Node """
            return self._next

        def setNext(self,x):
            """ Set the Previous Node """
            if (type(x) == ListNode):
                self._next = x
            else:
                raise TypeError("Must be of type 'ListNode'")
            return self

        def coupleToNext(self,other):
            """ Couple other ListNode as Next """
            if (type(other) == ListNode):
                self._next = other
                other.setPrev(self)
            else:
                raise TypeError("Next must be of type ListNode")
            return self
            
        def coupleToNext(self,other):
            """ Couple other ListNode as Next """
            if (type(other) == ListNode):
                self._prev = other
                other.setNext(self)
            else:
                raise TypeError("Next must be of type ListNode")
            return self 

    # End ListNode Defintion

    @staticmethod
    def coupleNodes(left,right):
        """ Couple Nodes s.t. 
            left->next = right &
            right->prev = left """
        left.setNext(right)
        right.setPrev(left)
        return True

    @staticMethod
    def joinThree(left,middle,right):
        """ Join Three Nodes such that
        left->middle->right &
        left<-middle<-right """
        left.setNext(middle)
        middle.setNext(right)
        right.setPrev(middle)
        middle.setPrev(left)
        return True

    """ Getters and Setters """

    def getHead(self):
        """ Get the HeadNode of this DoubleLinkedList """
        return self._head

    def getTail(self):
        """ GEt the TailNode of this DoubleLinkedList """
        return self._tail

    """ Public Interface """

    def append(self,data):
        """ Add a ListNode w/ data to the end of the linked List """
        newNode = DoubleLinkedList.ListNode(data)
        if (self._size == 0):       # Empty list
            DoubleLinkedList.coupleNodes(self._head,newNode)
            DoubleLinkedList.coupleNodes(newNode,self._prev)
        else:                       # Not empty List
            oldTailPrev = self._tail.getPrev()
            DoubleLinkedList.coupleNodes(oldTailPrev,newNode)
            DoubleLinkedList.coupleNodes(newNode,self._tail)
        self._size += 1
        return self

    def prepend(self,data):
        """ Add a ListNode w/ data to the start of the linked list """
        newNode = DoubleLinkedList.ListNode(data)
        if (self._size == 0):       # Empty list
            DoubleLinkedList.coupleNodes(self._head,newNode)
            DoubleLinkedList.coupleNodes(newNode,self._prev)
        else:
            oldHeadNext = self._head.getNext()
            DoubleLinkedList.coupleNodes(self._head,newNode)
            DoubleLinkedList.coupleNodes(newNode,oldHeadNext)
        self._size += 1
        return self

    def popTail(self):
        """ Remove and Return data ferom tail->prev """
        pass

    def popHead(self):
        """ Remove and return data from head->next """
        toRemove = self._head.getNext()
        DoubleLinkedList.coupleNodes(self._head,toRemove.getNext())
        return toRemove


    def insertAtIndex(self,data,index):
        """ Insert Node w/ data at index """
        if (index >= self._size or index < 0):
            raise IndexError("Index out of bounds")
        newNode = ListNode(data)
        raise NotImplementedError()

    def removeAtIndex(self,index):
        """ Remove Node at index, return data """
        if (index >= self._size or index < 0):
            raise IndexError("Index out of bounds")
        raise NotImplementedError()

    """ Magic Methods """

    def __iter__(self):
        """ Forward Iterator for DoubleLinkedList """
        currentNode = self._head
        while (currentNode != self.tail):
            yield currentNode.getData()
            currentNode = currentNode.getNext()
        return self._tail.getData()

    def __getitem__(self,index):
        """ Index Operator """
        if (index >= self._size or index < 0):
            raise IndexError("Index out of bounds")
        currentNode = self._head
        for i in range(index):
            currentNode = currentNode.getNext()
        return currentNode.getData()

    def __str__(self):
        """ String of DoubleLinkedList Instance """
        return "DoubleLinkedList w/ " + str(self._size) + " nodes"

    def __repr__(self):
        """ Programmer Representation of DoubleLinkedList Instance """
        return "DoubleLinkedList w/ " + str(self._size) + " nodes"

    def __len__(self):
        """ Get the Number of Non-sentinel Nodes in this Linked List """
        return self._size

class DirectedGraph:
    """
    DirectedGraph Type
        Represents the parent structure of a directed and unweighted graph
    --------------------------------

    --------------------------------
    """

    pass

class SimpleStack:
    """ Simple Stack Implementation """

    def __init__(self,cap,dType):
        """ Constructor for SimpleStack Instance """
        self._top   = -1
        self._dType = dType
        self._cap   = cap
        self._arr   = np.empty(shape=(self._cap),dtype=self._dType)

    def __del__(self):
        """ Destructor for SimpleStack Instance """
        pass

    """ Public Interface """

    def empty(self):
        """ Get T/F if Stack is Empty """
        return (self._top == -1)

    def size(self):
        """ Get the number of items in the stack """
        return self._top + 1

    def capacity(self):
        """ Get Current Capacity of the Stack """
        return self._cap

    def top(self):
        """ Get top of Stack """
        if (self.empty()):  # stack is empty
            raise ValueError("Cannot Get Top of Empty Stack")
        return self._arr[self._top]

    def push(self,x):
        """ Push an Element to Top of the Stack """
        if (self.size() == self.capacity()):
            # Need to Expand the Size of the Stack
            self.expand()
            self.expand()
        self._top += 1
        self._arr[self._top] = x
        return self

    def pop(self):
        """ Pop and element from the top of the stack """
        if (self.empty()):  # stack is empty
            raise ValueError("Cannot Pop From Empty Stack")
        else:
            self._top -= 1
        return self

    """ Private Interface """

    def expand(self):
        """ Double the Size of the Current Stack """
        additon = np.empty(shape=(self._cap),dtype=self._dType)
        self._arr = np.append(self._arr,additon)
        return self

    """ Magic Methods """

    def __str__(self):
        """ Return String representation of Stack """
        return str(self._arr)

    def __repr__(self):
        """ Return REPR representation of instance """
        return "Stack w/ (" + self._top + "/" + self._cap + ") spots filled"
    
    def __len__(self):
        """ Get the Length of the stack """
        return self._top + 1