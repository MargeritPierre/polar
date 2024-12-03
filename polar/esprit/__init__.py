# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:05:02 2023

@author: margeritadm
"""

import numpy as np
import scipy as sp

#--------------------------------------------------------
# ESPRIT FUNCTION
#--------------------------------------------------------
def esprit(signal, axes=None, R=None, smoothing_ratio=0.5, decim=1, fcn=None, solver=None):
    esp = Esprit(signal, axes, R, smoothing_ratio, decim, fcn, solver)
    esp.signalSubspace()
    esp.spectralMatrix()
    esp.poles()
    esp.wavenumbers()
    return esp


#--------------------------------------------------------
# ESPRIT CLASS
#--------------------------------------------------------
class Esprit():
    """The class implementing the ESPRIT algorithm
    """
    def __init__(self, signal=None, axes=None, R=None, smoothing_ratio=0.5, decim=1, fcn=None, solver=None):
        """Class constructor"""
        self.Signal = np.array(signal) # the input signal (np.array)
        # Axes along which to extract the wavenumbers/frequencies
        if axes is None: axes = np.arange(self.Signal.ndim) # all dimensions by default
        self.Axes = np.array(axes).flatten()
        # Signal shape
        self.L = np.array(self.Signal.shape).flatten()
        # Signal decimation along wavenuber dimensions (integers>=1)
        self.Decimation = self.L*[0] + 1 # no decimation by default
        decim = np.array(decim).flatten()
        if decim.size==1 or decim.size==self.Axes.size: self.Decimation[self.Axes] = decim
        else: raise NotImplemented # TODO decimation of signal snapshots ?
        # Family of functions to estimate ("exp" or "cos")
        if fcn is not None: raise NotImplementedError("Only exponential functions are implemented for now") # TODO
        self.Function = fcn
        # Spatial smoothing ratio along wavenumber axes (0<=N/L<=1)
        self.SmoothingRatio = float(smoothing_ratio)
        # Size of signals "snapshots"
        self.M = self.L*[0] + 1 # one by default on every signal axes
        self.M[self.Axes] = np.floor(self.L[self.Axes]/self.Decimation*(1.0-self.SmoothingRatio)+1) 
        # Number of signal "snapshots" or "realizations"
        self.N = self.L.copy() # full length by default on every signal axes
        self.N[self.Axes] = self.L[self.Axes] - (self.M[self.Axes]-1)*self.Decimation
        # Signal order candidates
        if R is None: R = np.arange(np.prod(self.M)-1) + 1
        else: R = np.array(R).flatten()
        R = R[R<np.prod(self.M)]
        self.R = R
        # signal subspace extraction ("eig" or "eigs")
        if solver is None: solver = "eig" # TODO - solver choice criterion
        self.Solver = solver

    def signalSubspace(self):
        """Extract the signal subspace"""
        if self.Solver=="eig": # Covariance-based eigenvalue decomposition
            # Block-hankel matrix indices
            from .. import hankel
            iH = hankel.blockHankelMatrixIdx(self.L,self.M,self.Decimation)
            H = np.matrix(self.Signal.flatten()[iH])
            # Covariance
            Css = H @ H.H
            # Signal subspace
            self.sigma, self.W = sp.linalg.eigh(Css)
            # Sort the eigenvectors
            idx = np.argsort(np.abs(self.sigma))[::-1]
            Rmax = np.max(self.R)
            self.sigma = self.sigma[idx[:Rmax]]
            self.W = self.W[:,idx[:Rmax]]
        elif self.Solver=="eigs":
            raise NotImplementedError # TODO
        return self.W

    def spectralMatrix(self,R=None):
        """Compute the spectral matri.x.ces"""
        # signal order infos
        if R is None: R=self.R
        nR = R.size
        Rmax = np.max(R)
        # reshape the signal subspace as a D+1 tensor
        W = np.reshape(np.array(self.W),np.append(self.M,Rmax))
        # list spectral matrices: list of size (Rmax,nDims) 
        self.F = [[np.NaN for ax in range(self.L.size)] for r in range(Rmax)]
        for ax in self.Axes:
            # Shifted signal subspaces
            idx = np.arange(self.M[ax]-1)
            Wup = np.matrix(np.reshape(np.take(W,idx,ax),(-1,Rmax)))
            Wdwn = np.matrix(np.reshape(np.take(W,idx+1,ax),(-1,Rmax)))
            if self.Axes.size==1: # Indirect computation (rank-one modification)
                PI = Wup.H @ Wdwn
                w = np.matrix(np.reshape(np.take(W,self.M[ax]-1,ax),(Rmax,1)))
                iw2 = 1/(1-np.cumsum(np.power(np.abs(w),2)))
                for r in R:
                    Fr = PI[:r,:r]
                    Fr = Fr + (iw2[0,r-1] * w[:r,0]) @ (w[:r,0].H @ Fr)
                    self.F[r-1][ax] = Fr
            else: # Indirect computation (rank-one modification)
                for r in R:
                    self.F[r-1][ax] = sp.linalg.lstsq(Wup,Wdwn)[0]
        return self.F

    def poles(self,R=None,beta=None):
        """Extract the signal poles"""
        if R is None: R=self.R
        Rmax = np.max(R)
        if beta is None: beta = np.power(1e-12,np.arange(self.Axes.size))
        # one list of pole for each signal candidate: 
        self.Z = [[] for r in range(Rmax)]
        for r in R:
            Fr = self.F[r-1][:]
            F = sum([Fr[ax]*beta[ax] for ax in self.Axes])
            z,T = sp.linalg.eig(F)
            self.Z[r-1] = np.ones((r,self.L.size))*(1+0j)
            for ax in self.Axes: # Z = T^{-1} . F . T
                self.Z[r-1][:,ax] = np.diag( np.linalg.solve( T , Fr[ax] @ T ) )
        return self.Z

    def wavenumbers(self,R=None):
        """Extract the signal wavenumbers"""
        if R is None: R=self.R
        self.k = [-1j*np.log(self.Z[r-1]) for r in R]
        return self.k

