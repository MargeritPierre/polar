# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:05:02 2023

@author: margeritadm
"""

import numpy as np
import xarray as xr


#--------------------------------------------------------
# SIGNAL CLASS
#--------------------------------------------------------
class Signal(xr.DataArray):
    """The general signal model containing signal data to be processed
    It adds some signal processing specific methods to the xarray.DataArray class.
    """

    __slots__ = ()

    # import class methods
    from ._fourier import fft, ifft, fftshift, ifftshift

    def _print(self):
        print(self)
