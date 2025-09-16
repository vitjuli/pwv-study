
import numpy as np
from scipy.signal import welch
from scipy.stats import gaussian_kde
from numpy.linalg import LinAlgError

def signal_quality_indices(x, fs):
    f,P=welch(x,fs=fs,nperseg=min(2048,len(x))); P=P+1e-18; band=(f>=0.5)&(f<=5.0); bp=np.trapz(P[band],f[band]); total=np.trapz(P,f); rest=max(total-bp,1e-18)
    snr=bp/rest; p=P/P.sum(); ent=-np.sum(p*np.log(p+1e-18)); return {"snr_proxy":float(snr),"spectral_entropy":float(ent)}

def gate_hit_ratio(valleys, anchors, fs, gate_ms):
    if anchors.size==0: return float('nan')
    g=max(1,int(gate_ms*fs/1000)); hits=0
    for a in anchors:
        i0=a; i1=a+g; hits+=int(np.any((valleys>=i0)&(valleys<i1)))
    return hits/anchors.size

def mad_filter(values, thr=3.5):
    v=values[np.isfinite(values)]
    if v.size==0: return np.zeros_like(values,bool)
    med=np.median(v); mad=np.median(np.abs(v-med))+1e-12; score=np.abs(values-med)/(1.4826*mad); return score<thr

def kde_mode(values):
    v=values[np.isfinite(values)]
    if v.size<3: return float(np.median(v)) if v.size else float('nan')
    try:
        kde=gaussian_kde(v)
        xs=np.linspace(np.min(v),np.max(v),256)
        ys=kde(xs)
        return float(xs[int(np.argmax(ys))])
    except Exception:
        return float(np.median(v))
