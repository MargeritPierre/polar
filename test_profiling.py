#%% ESPRIT 1D MONOCHROMATIC
import polar
import numpy as np
import scipy as sp

L = 1024
k = np.pi/5
x = np.arange(L)
s = np.exp(1j*k*x)


ke = polar.esprit.esprit(s,R=1).k[-1]
print("k =",k,"\nke =",ke,"\nerror =",np.linalg.norm(k-ke))