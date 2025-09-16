import numpy as np
from sklearn.cluster import KMeans
def beat_features(signal, anchors, fs, win_ms=300):
    w=max(1,int(win_ms*fs/1000)); feats=[]; centers=[]
    for a in anchors:
        i0=max(0,a-w//2); i1=min(signal.size,a+w//2); x=signal[i0:i1]
        if x.size<10: continue
        amp=x.max()-x.min(); slope=np.max(np.gradient(x)) if x.size>3 else 0.0; curv=np.min(np.gradient(np.gradient(x))) if x.size>5 else 0.0
        feats.append([amp,slope,curv,float(x.size)]); centers.append(a)
    if not feats: return np.zeros((0,4)), np.zeros((0,),int)
    return np.array(feats,float), np.array(centers,int)
def kmeans_select_stable(feats, centers, k=2, seed=0):
    if feats.shape[0]==0: return np.array([],int)
    k=min(k,feats.shape[0]); km=KMeans(n_clusters=k, random_state=seed, n_init=10); lab=km.fit_predict(feats); c=int(np.argmax(np.bincount(lab)))
    return centers[lab==c]
