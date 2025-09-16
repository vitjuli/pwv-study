import numpy as np
def compute_feature_lags(prox_idx, dist_idx, fs):
    if prox_idx.size==0 or dist_idx.size==0: return np.array([]), float('nan')
    lags=[]; j=0
    for i in range(len(prox_idx)):
        p=prox_idx[i]
        while j<len(dist_idx) and dist_idx[j]<p: j+=1
        if j>=len(dist_idx): break
        lags.append((dist_idx[j]-p)/fs); j+=1
    lags=np.array(lags,float); med=float(np.median(lags)) if lags.size else float('nan'); return lags, med
def velocity_from_lag(distance_m, lag_seconds):
    if not np.isfinite(lag_seconds) or lag_seconds<=0: return float('nan')
    return distance_m/lag_seconds
