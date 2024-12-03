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

# Figure initialization
fig = plt.figure(figsize=(15,8))
# Time
axT = fig.add_subplot(2,1,1)
plT = plt.plot(np.nan,np.nan,'k')[0]
# Frequency
axF = fig.add_subplot(2,1,2)
plF = plt.plot(np.nan,np.nan,'k')[0]
plE = plt.plot(np.nan,np.nan,'r')[0]
axF.set_xscale('log')
# axF.set_yscale('log')

def processFcn(evt):
    #print(np.array(arg.inputBuffer.getChannelData(0).to_py()).shape)
    # Time signal
    buffer = np.array(evt.inputBuffer.getChannelData(0).to_py())
    time = np.arange(buffer.size)/evt.inputBuffer.sampleRate + evt.playbackTime
    plT.set_xdata(time)
    plT.set_ydata(buffer)
    axT.set_xlim(time[0],time[-1])
    axT.set_ylim(-1,1)
    # Signal spectum
    spectrum = np.fft.rfft(buffer,norm='forward')
    frequency = np.arange(spectrum.size)*evt.inputBuffer.sampleRate/buffer.size
    plF.set_xdata(frequency)
    plF.set_ydata(np.abs(spectrum))
    axF.set_xlim(frequency[1],frequency[-1])
    axF.set_ylim(0.001,0.5)
    # Apply ESPRIT
    esp = polar.esprit.esprit(buffer,R=2)
    fe = esp.k[-1]*evt.inputBuffer.sampleRate/2/np.pi
    fe = fe[np.real(fe)>0]
    plE.set_xdata((fe[:,None]*[[1,1,np.nan]]).flatten())
    plE.set_ydata((fe[:,None]*0 + [[axF.get_ylim()[0],axF.get_ylim()[1],np.nan]]).flatten())
    # print(ke)
    display(fig, target="mpl",append=False)
    return 