# A 2D ESPRIT DEMO USING PYSCRIPT

import polar
import numpy as np
import matplotlib.pyplot as plt

L = np.array([10,15])
R = 3

k = 0.25*np.pi*(2*np.random.rand(R,L.size)-1)
a = np.random.randn(R) + 1j*np.random.randn(R)
xi = [np.arange(l) for l in L]
x = np.hstack([xi.flatten()[:,None] for xi in np.meshgrid(*xi, indexing='ij')])
s = np.reshape(np.sum(a[None,:]*np.exp(1j*np.sum(k.T[None,:,:]*x[:,:,None],axis=1)),axis=1),L)

dims = ["x"+ str(ax) for ax in range(L.size)] 
coords = [(dims[ax],xi[ax]) for ax in range(L.size)]
s = polar.signal.Signal(s,dims=dims,coords=coords)

fig = plt.figure()
np.real(s).plot()

"""

esp = polar.esprit.esprit(np.array(s),R=R)
ke = esp.k[-1]

dk = np.sqrt(np.sum(np.power(np.abs(k[:,None,:] - ke[None,:,:]),2),axis=2))
dk = np.min(dk,axis=0)

F = s.fft(dims).fftshift()

fig = plt.figure()
# fig.add_subplot(1,2,1)
# np.real(s).plot()
# fig.add_subplot(1,2,2)
# np.abs(F).plot()
plt.plot(k[:,1]/2/np.pi,k[:,0]/2/np.pi,'ok')
plt.plot(np.real(ke[:,1])/2/np.pi,np.real(ke[:,0])/2/np.pi,'+r')

print("k =",k,"\nke =",ke,"\nerror =",np.linalg.norm(dk))
"""
display(fig, target="mpl")