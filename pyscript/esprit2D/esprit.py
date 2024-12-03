# A 2D ESPRIT DEMO USING PYSCRIPT

# Imports
import polar
import numpy as np
#
import matplotlib
import matplotlib.pyplot as plt
#
from pyodide.ffi import create_proxy
import js

# The signal grid
L = np.array([4,6])*8
xi = [np.arange(l) for l in L]
x = np.hstack([xi.flatten()[:,None] for xi in np.meshgrid(*xi, indexing='ij')])
dims = ["x"+ str(ax) for ax in range(L.size)] 
coords = [(dims[ax],xi[ax]) for ax in range(L.size)]

# List of wavenumbers & amplitudes
def wavenumbers(R):
    return 0.40*np.pi*(2*np.random.rand(R,L.size)-1)
def amplitudes(R):
    return np.ones(R)#np.random.randn(R) + 1j*np.random.randn(R)

# Signal generation
def generate_signal(k,a):
    return np.reshape(np.sum(a[None,:]*np.exp(1j*np.sum(k.T[None,:,:]*x[:,:,None],axis=1)),axis=1),L)
def normalize(s):
    return (s-np.min(s.flatten())[None,None])/((np.max(s.flatten())-np.min(s.flatten()))[None,None])

# UI definition
SOrange = js.document.querySelector("#signal_order")
GenBtn = js.document.querySelector("#generate")
def on_range_update(event):
    label = event.currentTarget.nextElementSibling
    label.innerText = event.currentTarget.value
    update()
def on_button_update(event): update()
range_proxy = create_proxy(on_range_update)
SOrange.addEventListener("input", range_proxy)
button_proxy = create_proxy(on_button_update)
GenBtn.addEventListener("click", button_proxy)

# Display
# initialization
fig = plt.figure(figsize=(15,6))
fig.add_subplot(1,2,1)
s = polar.signal.Signal(np.random.rand(*L),dims=dims,coords=coords)
srf_s = s.plot()
plt.title("Signal")
fig.add_subplot(1,2,2)
srf_fft = (np.abs(s.fft(dims).fftshift())/np.prod(L)).plot()
plK = plt.plot(np.nan,np.nan,'ok')[0]
plKe = plt.plot(np.nan,np.nan,'+r')[0]
plt.title("Wavenumber spectrum")
# update
def update():
    # retrieve UI values
    R = int(SOrange.value)
    # generate the signal
    k = wavenumbers(R)
    a = amplitudes(R)
    s = polar.signal.Signal(generate_signal(k,a),dims=dims,coords=coords)
    F = s.fft(dims).fftshift()
    # apply ESPRIT
    esp = polar.esprit.esprit(np.array(s),R=R)
    ke = esp.k[-1]
    # update plot data
    srf_s.set_array(normalize(np.array(np.real(s))))
    srf_fft.set_array(normalize(np.array(np.abs(F))))
    plK.set_xdata(k[:,1]/2/np.pi)
    plK.set_ydata(k[:,0]/2/np.pi)
    plKe.set_xdata(np.real(ke[:,1])/2/np.pi)
    plKe.set_ydata(np.real(ke[:,0])/2/np.pi)
    display(fig, target="mpl",append=False)


# first update
update()