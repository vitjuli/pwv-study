import numpy as np
def _lognormal_pulse(t,t0,amp,width,skew):
    x=(t-t0)/max(width,1e-6); x=np.clip(x,1e-9,None)
    y=amp*np.exp(-0.5*(np.log(x)/skew)**2)/(x*np.sqrt(2*np.pi)*abs(skew)); y[t<t0]=0; return y
def synthesize_ppg_bundle(seconds=12, fs=1000, hr=70, hrv=0.02, noise_snr_db=15, motion_steps=0, seed=1):
    rng=np.random.default_rng(seed); n=int(seconds*fs); t=np.arange(n)/fs
    hr_series=hr*(1.0+hrv*np.cumsum(rng.normal(0,0.005,size=n))); hr_series=np.clip(hr_series,35,180)
    inst=60.0/hr_series; beats=[0.3]
    while beats[-1]<seconds:
        idx=int(min(len(inst)-1, beats[-1]*fs)); beats.append(beats[-1]+float(inst[idx]))
    beats=np.array(beats[1:]); clean=np.zeros_like(t)
    for bt in beats: clean+=_lognormal_pulse(t,bt,1.0,0.15,0.35)
    if clean.ptp()>0: clean=(clean-clean.min())/clean.ptp()
    x=clean.copy(); p=np.mean(x**2)+1e-12; p_noise=p/(10**(noise_snr_db/10)); x+=rng.normal(0,np.sqrt(p_noise),size=x.size)
    for _ in range(max(0,motion_steps)):
        i=rng.integers(int(0.1*n),int(0.9*n)); x[i:]+=rng.choice([-1,1])*0.2*rng.uniform(0.4,1.2)
    fd=np.gradient(x)*fs; sd=np.gradient(fd)*fs
    return {"t":t,"ppg":x,"clean":clean,"fd":fd,"sd":sd}
