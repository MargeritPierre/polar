#%% DEMO FILE FOR THE POLAR TOOLBOX

#%% Generate a signal and compute its FFT
import numpy as np
import matplotlib.pyplot as plt
import polar.signal

t = np.arange(101)*.5 # time vector
x = np.arange(51)*.5 # spatial coordinates

data = np.random.randn(t.size,x.size)
data = np.exp(2j*(1+0.05j)*(np.random.rand()*t[:,np.newaxis]-np.random.rand()*x[np.newaxis,:]))

s = polar.signal.Signal(data,
                        dims=('t','x'),
                        coords=[("t",t),("x",x)]
                        )

F = s.fft(('t','x')).fftshift()

fig = plt.figure()
fig.add_subplot(1,2,1)
np.real(s).plot()
fig.add_subplot(1,2,2)
np.abs(F).plot()

# %% HANKEL MATRICES
import polar
import numpy as np

L = np.array([5,4]) 
M = None #np.floor(L/2)
D = 1

Ls,Ms,Ns,Ds = polar.hankel.hankelShapes(L,M,D)
print("L:",Ls,"M:",Ms,"N:",Ns,"D:",Ds)

iH = polar.hankel.blockHankelMatrixIdx(L,M,D)
print("iH:\n",iH)

#%% ESPRIT 1D MONOCHROMATIC
import polar
import numpy as np

L = 10
k = np.pi/5
x = np.arange(L)
s = np.exp(1j*k*x)

ke = polar.esprit.esprit(s,R=1).k[-1]
print("k =",k,"\nke =",ke,"\nerror =",np.linalg.norm(k-ke))

#%% ESPRIT 1D MULTICOMPONENT
import polar
import numpy as np
import matplotlib.pyplot as plt

L = 200
k = np.pi/np.array([10,-5])
a = np.random.randn(k.size) + 1j*np.random.randn(k.size)
x = np.arange(L)
s = np.sum(a[None,:]*np.exp(1j*k[None,:]*x[:,None]),axis=1)

esp = polar.esprit.esprit(s,R=k.size)
ke = esp.k[-1]

plt.plot(np.real(s))
plt.plot(np.imag(s))
print("k =",k,"\nke =",ke,"\nerror =",np.linalg.norm(np.sort(k.flatten())-np.sort(ke.flatten())))


#%% ESPRIT ND MULTICOMPONENT
import polar
import numpy as np
import matplotlib.pyplot as plt

L = np.array([40,60])
R = 10

k = 0.25*np.pi*(2*np.random.rand(R,L.size)-1)
a = np.random.randn(R) + 1j*np.random.randn(R)
xi = [np.arange(l) for l in L]
x = np.hstack([xi.flatten()[:,None] for xi in np.meshgrid(*xi, indexing='ij')])
s = np.reshape(np.sum(a[None,:]*np.exp(1j*np.sum(k.T[None,:,:]*x[:,:,None],axis=1)),axis=1),L)

dims = ["x"+ str(ax) for ax in range(L.size)] 
coords = [(dims[ax],xi[ax]) for ax in range(L.size)]
s = polar.signal.Signal(s,dims=dims,coords=coords)

esp = polar.esprit.esprit(np.array(s),R=R)
ke = esp.k[-1]

dk = np.sqrt(np.sum(np.power(np.abs(k[:,None,:] - ke[None,:,:]),2),axis=2))
dk = np.min(dk,axis=0)

F = s.fft(dims).fftshift()

fig = plt.figure()
fig.add_subplot(1,2,1)
np.real(s).plot()
fig.add_subplot(1,2,2)
np.abs(F).plot()
plt.plot(k[:,1]/2/np.pi,k[:,0]/2/np.pi,'ok')
plt.plot(np.real(ke[:,1])/2/np.pi,np.real(ke[:,0])/2/np.pi,'+r')
print("k =",k,"\nke =",ke,"\nerror =",np.linalg.norm(dk))

# %%
