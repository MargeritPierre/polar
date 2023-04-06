# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:05:02 2023

@author: margeritadm
"""

import numpy as np
import scipy as sp

def hankelShapes(L,M=None,D=1):
    """parse input shapes
    L: signal shape
    M: snapshot shape
    D: decimation factor.s
    """
    L = np.array(L).flatten()
    D = np.array(D).flatten() + L*[0]
    if M is None: M = np.ceil((L+D)/(D+1))
    M = np.array(M).flatten() + L*[0]
    N = L-(M-1)*D
    return L,M,N,D


def hankelMatrixIdx(L,M=None,D=1):
    """indices of a Hankel matrix
    L: signal shape
    M: snapshot shape
    D: decimation factor.s
    """
    L,M,N,D = hankelShapes(L,M,D)
    return np.arange(M)[:,np.newaxis]*D + np.arange(N)[np.newaxis,:] 

def hankelTensorIdx(L,M=None,D=1):
    """indices of a Hankel tensor
    L: signal shape
    M: snapshot shape
    D: decimation factor.s
    """
    L,M,N,D = hankelShapes(L,M,D)
    # Indices
    iH = 0
    for dim in range(L.size):
        # Hankel matrix indices associated to the current dimension
        iHd = hankelMatrixIdx(L[dim],M[dim],D[dim])
        # Add to the block-hankel tensor
        iHd = np.expand_dims(iHd,tuple(np.arange(2,2*L.size)))
        iH = iH*L[dim] + np.moveaxis(iHd,(0,1),(dim,L.size+dim))
    return iH

def blockHankelMatrixIdx(L,M=None,D=1):
    """indices of a block-hankel matrix
    L: signal shape
    M: snapshot shape
    D: decimation factor.s
    """
    L,M,N,D = hankelShapes(L,M,D)
    H = np.reshape(hankelTensorIdx(L,M,D),(int(np.prod(M)),-1))
    return H