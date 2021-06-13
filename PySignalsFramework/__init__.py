"""
PySignalsFramework
__init__ 
"""

import AudioTools
import LayersCustom
import LayersStandard
import LayerChain
import EffectsSystem

__version__ = "0.0.0"

def setmodule(module):
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