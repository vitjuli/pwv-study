import numpy as np, pandas as pd
from .preprocess import lowpass_fir
from .fiducials import detect_slope_rise_points, detect_curvature_valleys, gaussian_weighted_valley
from .clustering import beat_features, kmeans_select_stable
from .quality import signal_quality_indices, gate_hit_ratio, mad_filter, kde_mode
from .metrics import compute_feature_lags, velocity_from_lag
def _per_beat_table(lags, prox_idx, dist_idx, fs, pipeline):
    n=min(len(lags),len(prox_idx),len(dist_idx))
    return pd.DataFrame({"pipeline":[pipeline]*n,"prox_idx":prox_idx[:n],"dist_idx":dist_idx[:n],"lag_s":lags[:n]})
class WindowAlignedPipeline:
    def __init__(self, fs, cutoff=10.0, gate_ms=180): self.fs,self.cutoff,self.gate_ms=fs,cutoff,gate_ms
    def run(self, proximal, distal, distance_m, export_beats=False):
        x_p=lowpass_fir(proximal,self.fs,self.cutoff); x_d=lowpass_fir(distal,self.fs,self.cutoff)
        fd_p=np.gradient(x_p)*self.fs; fd_d=np.gradient(x_d)*self.fs; sd_p=np.gradient(fd_p)*self.fs; sd_d=np.gradient(fd_d)*self.fs
        rp=detect_slope_rise_points(fd_p,self.fs,0.35,300); rd=detect_slope_rise_points(fd_d,self.fs,0.35,300)
        vp=detect_curvature_valleys(sd_p,rp,self.fs,self.gate_ms); vd=detect_curvature_valleys(sd_d,rd,self.fs,self.gate_ms)
        lags,med=compute_feature_lags(vp,vd,self.fs); mask=mad_filter(lags,3.0); lags_rob=lags[mask]
        med_rob=float(np.median(lags_rob)) if lags_rob.size else float('nan'); mode=kde_mode(lags) if lags.size else float('nan')
        vel=velocity_from_lag(distance_m, med_rob if np.isfinite(med_rob) else med)
        return {"lag_median":float(med),"lag_median_robust":float(med_rob),"lag_mode":float(mode),"velocity":float(vel),
                "n_pairs":int(lags.size),"n_pairs_robust":int(lags_rob.size)}, (_per_beat_table(lags,vp,vd,self.fs,"WindowAligned") if export_beats else None)
class FeatureLagPipeline:
    def __init__(self, fs, cutoff=10.0): self.fs,self.cutoff=fs,cutoff
    def run(self, proximal, distal, distance_m, export_beats=False):
        x_p=lowpass_fir(proximal,self.fs,self.cutoff); x_d=lowpass_fir(distal,self.fs,self.cutoff)
        fd_p=np.gradient(x_p)*self.fs; fd_d=np.gradient(x_d)*self.fs
        rp=detect_slope_rise_points(fd_p,self.fs,0.4,280); rd=detect_slope_rise_points(fd_d,self.fs,0.4,280)
        lags,med=compute_feature_lags(rp,rd,self.fs); mask=mad_filter(lags,3.0); lags_rob=lags[mask]
        med_rob=float(np.median(lags_rob)) if lags_rob.size else float('nan'); mode=kde_mode(lags) if lags.size else float('nan')
        vel=velocity_from_lag(distance_m, med_rob if np.isfinite(med_rob) else med)
        return {"lag_median":float(med),"lag_median_robust":float(med_rob),"lag_mode":float(mode),"velocity":float(vel),
                "n_pairs":int(lags.size),"n_pairs_robust":int(lags_rob.size)}, (_per_beat_table(lags,rp,rd,self.fs,"FeatureLag") if export_beats else None)
class HybridMedianPipeline:
    def __init__(self, fs, cutoff=10.0): self.fs,self.cutoff=fs,cutoff
    def run(self, proximal, distal, distance_m, export_beats=False):
        s1,b1=WindowAlignedPipeline(self.fs,self.cutoff).run(proximal,distal,distance_m,export_beats)
        s2,b2=FeatureLagPipeline(self.fs,self.cutoff).run(proximal,distal,distance_m,export_beats)
        meds=[x for x in (s1.get("lag_median_robust"), s2.get("lag_median_robust"), s1.get("lag_mode"), s2.get("lag_mode")) if (x==x)]
        med=float(np.median(meds)) if meds else float('nan'); vel=velocity_from_lag(distance_m,med)
        beats=None
        if export_beats and (b1 is not None or b2 is not None):
            import pandas as pd; beats=pd.concat([b for b in (b1,b2) if b is not None], ignore_index=True); beats["pipeline_combo"]="HybridMedian"
        return {"lag_median":float(med),"velocity":float(vel),"n_pairs":int(max(s1["n_pairs_robust"], s2["n_pairs_robust"]))}, beats
class ClusterGuidedPipeline:
    def __init__(self, fs, cutoff=10.0, sigma_ms=40, gate_ms=180, k=2, seed=0):
        self.fs,self.cutoff,self.sigma_ms,self.gate_ms,self.k,self.seed=fs,cutoff,sigma_ms,gate_ms,k,seed
    def run(self, proximal, distal, distance_m, export_beats=False):
        x_p=lowpass_fir(proximal,self.fs,self.cutoff); x_d=lowpass_fir(distal,self.fs,self.cutoff)
        fd_p=np.gradient(x_p)*self.fs; fd_d=np.gradient(x_d)*self.fs; sd_p=np.gradient(fd_p)*self.fs; sd_d=np.gradient(fd_d)*self.fs
        rp=detect_slope_rise_points(fd_p,self.fs,0.35,300); rd=detect_slope_rise_points(fd_d,self.fs,0.35,300)
        vp0=detect_curvature_valleys(sd_p,rp,self.fs,self.gate_ms); vd0=detect_curvature_valleys(sd_d,rd,self.fs,self.gate_ms)
        vp=np.array([gaussian_weighted_valley(sd_p,c,self.fs,self.sigma_ms) for c in vp0],int)
        vd=np.array([gaussian_weighted_valley(sd_d,c,self.fs,self.sigma_ms) for c in vd0],int)
        fp,cp=beat_features(x_p,rp,self.fs,300); fd_feats,cd=beat_features(x_d,rd,self.fs,300)
        sp=kmeans_select_stable(fp,cp,self.k,self.seed); sd_=kmeans_select_stable(fd_feats,cd,self.k,self.seed)
        vp_sel=np.array([v for v in vp if any(abs(v-s)<=int(0.15*self.fs) for s in sp)],int)
        vd_sel=np.array([v for v in vd if any(abs(v-s)<=int(0.15*self.fs) for s in sd_)],int)
        lags,med=compute_feature_lags(vp_sel,vd_sel,self.fs); mask=mad_filter(lags,3.0); lags_rob=lags[mask]
        med_rob=float(np.median(lags_rob)) if lags_rob.size else float('nan'); mode=kde_mode(lags) if lags.size else float('nan')
        vel=velocity_from_lag(distance_m, med_rob if np.isfinite(med_rob) else med)
        return {"lag_median":float(med),"lag_median_robust":float(med_rob),"lag_mode":float(mode),"velocity":float(vel),
                "n_pairs":int(lags.size),"n_pairs_robust":int(lags_rob.size)}, (_per_beat_table(lags,vp_sel,vd_sel,self.fs,"ClusterGuided") if export_beats else None)
