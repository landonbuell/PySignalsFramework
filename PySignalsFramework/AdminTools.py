"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           AdminTools.py
Description:
"""

from __init__ import *

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

        #### CLASS DEFINTIONS ####

class SimpleStack:
    """ Simple Stack Implementation """

    def __init__(self,cap,dType):
        """ Constructor for SimpleStack Instance """
        self._top = -1
        self._dType = dType
        self._cap = cap
        self._arr = np.empty(shape=(self._cap),dtype=self._dType)

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