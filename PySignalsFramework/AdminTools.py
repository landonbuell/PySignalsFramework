"""
Author:         Landon Buell
Date:           August 2021
Solution:       PySignalFramework
Project:        PySignalFramework
File:           AdminTools.py
Description:
"""

def setModule(module):
    """
    Decorator for overriding __module__ on a function or class.

    Example usage::

        @setmodule('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
