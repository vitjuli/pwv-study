import numpy as np
from scipy.signal import firwin, lfilter
def lowpass_fir(x, fs, cutoff=10.0, taps=129):
    b=firwin(taps, cutoff=cutoff, fs=fs); return lfilter(b,[1.0],x)
