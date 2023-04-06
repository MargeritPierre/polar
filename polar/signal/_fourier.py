# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:05:02 2023

@author: margeritadm
"""

import numpy as np
import xarray as xr


# Standard Fourier coordinates transformations ==========================================================
_fourierCoordNames = {
    "t" : "f",
    "time" : "frequency",
    "x" : "mu",
    "space" : "wavenumber",
}

# Fourier Transform ==========================================================
def _transform(self,n=None,dims=None,isinverse=False): 
    if dims is None: dims = self.dims[-1]
    # make a copy of the array
    s = self.copy()
    # apply the transform
    ax = self.get_axis_num(tuple(dims))
    if isinverse: s.values = np.fft.ifftn(self.values, s=n, axes=ax)
    else: s.values = np.fft.fftn(self.values, s=n, axes=ax)
    # change the coordinate values
    for dim in dims:
        if dim in s.coords.keys(): 
            dc = np.diff(s.coords[dim])
            if all(np.abs(dc-dc[0])<1e3*np.finfo(float).eps): 
                L = s.coords[dim][-1]-s.coords[dim][0]+dc[0]
                s.coords[dim] = np.arange(s.coords[dim].size)/L.to_numpy()
    # change the dimension name
    for dim in dims:
        if isinverse:
            if dim in _fourierCoordNames.values(): 
                newdim = [k for k,v in _fourierCoordNames.items() if v==dim][0]
            else: newdim = "ift_" + dim
        else:
            if dim in _fourierCoordNames.keys(): 
                newdim = _fourierCoordNames[dim]
            else: newdim = "ft_" + dim
        s = s.rename({dim : newdim})   
    # return
    return s

def fft(self,dims=None,n=None): 
    return _transform(self,n,dims,isinverse=False)

def ifft(self,dims=None,n=None): 
    return _transform(self,n,dims,isinverse=True)


# Fourier Shift ==========================================================
def _shift(self,dims=None,isinverse=False): 
    if dims is None: dims = self.dims
    # make a copy of the array
    s = self.copy()
    # apply the transform
    ax = self.get_axis_num(tuple(dims))
    if isinverse: s.values = np.fft.ifftshift(self.values, axes=ax)
    else: s.values = np.fft.fftshift(self.values, axes=ax)
    # change the coordinates values
    for dim in dims:
        if dim in s.coords.keys(): 
            if isinverse: c0 = s.coords[dim][0]
            else: c0 = s.coords[dim][int(s.coords[dim].size/2)]
            s.coords[dim] = s.coords[dim] - c0
    # return
    return s

def fftshift(self,dims=None): 
    return _shift(self,dims=dims,isinverse=False)

def ifftshift(self,dims=None): 
    return _shift(self,dims=dims,isinverse=True)