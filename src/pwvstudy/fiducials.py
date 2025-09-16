import numpy as np
from scipy.signal import find_peaks
def detect_slope_rise_points(fd, fs, rel_height=0.3, min_distance_ms=300):
    h=fd.min()+rel_height*(fd.max()-fd.min()); d=max(1,int(min_distance_ms*fs/1000)); p,_=find_peaks(fd,height=h,distance=d); return p
def detect_curvature_valleys(sd, anchors, fs, gate_ms=180):
    out=[]; g=max(1,int(gate_ms*fs/1000))
    for a in anchors:
        i0=a; i1=min(sd.size,a+g)
        if i0>=i1: continue
        out.append(i0+int(np.argmin(sd[i0:i1])))
    return np.array(out,dtype=int) if out else np.array([],dtype=int)
def gaussian_weighted_valley(sd, center, fs, sigma_ms=40, half_window_ms=120):
    hw=max(1,int(half_window_ms*fs/1000)); i0=max(0,center-hw); i1=min(sd.size,center+hw); x=sd[i0:i1].astype(float)
    if x.size<=3: return center
    sigma=max(1.0,sigma_ms*fs/1000.0); idx=np.arange(x.size)-(x.size-1)/2; w=np.exp(-0.5*(idx/sigma)**2); score=w*(-x)
    return i0+int(np.argmax(score))
