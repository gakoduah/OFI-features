"""
Advanced analyses that go beyond estimation:
  1. Cross-asset lead-lag: at what lag does one stock's OFI predict another's return?
  2. Spectral structure of the contemporaneous return/impact system (market mode).
  3. Economic significance: does the cross-asset OFI signal survive transaction costs?
  4. Sub-sample stability (morning vs afternoon).
Real LOBSTER data, five same-day stocks.
Run: python3 ofi_advanced.py data_dir
"""
from __future__ import annotations
import sys, glob, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LEVELS=10; BUCKET=1.0; OUT="outputs"
TICKERS=["AAPL","AMZN","GOOG","INTC","MSFT"]

def load(tk, D):
    mp=glob.glob(f"{D}/{tk}_*_message_10.csv")[0]; bp=glob.glob(f"{D}/{tk}_*_orderbook_10.csv")[0]
    msg=pd.read_csv(mp,header=None,names=["time","ty","oid","sz","px","dr"])
    book=pd.read_csv(bp,header=None).values.astype(float); t=msg["time"].values
    def lvl(m): o=4*(m-1); return book[:,o],book[:,o+1],book[:,o+2],book[:,o+3]
    def ofi(m):
        ap,asz,bp2,bsz=lvl(m)
        dWb=np.where(bp2>np.roll(bp2,1),bsz,np.where(bp2<np.roll(bp2,1),-np.roll(bsz,1),bsz-np.roll(bsz,1)))
        dWa=np.where(ap<np.roll(ap,1),asz,np.where(ap>np.roll(ap,1),-np.roll(asz,1),asz-np.roll(asz,1)))
        e=dWb-dWa; e[0]=0; return e
    E=np.column_stack([ofi(m) for m in range(1,11)])
    a1,as1,b1,bs1=lvl(1); mid=(a1+b1)/2/1e4; spr=(a1-b1)/1e4
    bk=np.floor(t/BUCKET).astype(int)
    df=pd.DataFrame(E,columns=[f"o{m}" for m in range(1,11)]); df["bk"]=bk; df["mid"]=mid; df["spr"]=spr; df["sod"]=t
    g=df.groupby("bk").agg({**{f"o{m}":"sum" for m in range(1,11)},"mid":["first","last"],"spr":"mean","sod":"first"})
    g.columns=[f"o{m}" for m in range(1,11)]+["mf","ml","spr","sod"]; g=g.reset_index().iloc[1:-1]
    g["ret"]=(g["ml"]-g["mf"])/g["mf"]*1e4  # bps
    g["sprbps"]=g["spr"]/g["mf"]*1e4
    Z=g[[f"o{m}" for m in range(1,11)]].values; Zs=(Z-Z.mean(0))/Z.std(0)
    w,V=np.linalg.eigh(np.cov(Zs.T)); pc1=V[:,-1]; pc1=pc1 if pc1.mean()>=0 else -pc1
    g["ofi"]=Zs@pc1
    return g.set_index("bk")

D=sys.argv[1] if len(sys.argv)>1 else "xi_data"
S={tk:load(tk,D) for tk in TICKERS}
ofi=pd.concat({tk:S[tk]["ofi"] for tk in TICKERS},axis=1).dropna()
ret=pd.concat({tk:S[tk]["ret"] for tk in TICKERS},axis=1).dropna()
spr=pd.concat({tk:S[tk]["sprbps"] for tk in TICKERS},axis=1).dropna()
sod=S["AAPL"]["sod"]
idx=ofi.index.intersection(ret.index)
ofi=ofi.loc[idx]; ret=ret.loc[idx]; spr=spr.loc[idx]
ofi_s=(ofi-ofi.mean())/ofi.std(); ret_s=(ret-ret.mean())/ret.std()
n=len(TICKERS); res={"common":int(len(idx))}

# ---- 1) lead-lag cross-correlation ----
lags=range(-5,6)
# average cross (i!=j) correlation of OFI_j(t-l) with ret_i(t)
cc_cross=[]; cc_own=[]
for l in lags:
    cross=[]; own=[]
    for i,ti in enumerate(TICKERS):
        yi=ret_s[ti].values
        for j,tj in enumerate(TICKERS):
            x=np.roll(ofi_s[tj].values,l)  # OFI_j shifted; l>0 -> past OFI predicts current ret
            s=slice(max(l,0),len(yi)+min(l,0))
            c=np.corrcoef(x[s],yi[s])[0,1]
            (own if i==j else cross).append(c)
    cc_cross.append(np.mean(cross)); cc_own.append(np.mean(own))
res["leadlag_lags"]=list(lags); res["leadlag_cross"]=cc_cross; res["leadlag_own"]=cc_own
fig,ax=plt.subplots(figsize=(6.5,4))
ax.axvline(0,color="gray",ls=":"); ax.axhline(0,color="gray",lw=.5)
ax.plot(list(lags),cc_own,"o-",label="own (i=j)")
ax.plot(list(lags),cc_cross,"s-",label="cross (i$\\neq$j)")
ax.set_xlabel("lag $\\ell$: OFI$_j(t-\\ell)$ vs ret$_i(t)$   (>0 = OFI leads return)")
ax.set_ylabel("avg correlation"); ax.set_title("Lead-lag: own impact is instantaneous, cross is at short lag")
ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(f"{OUT}/fig13_leadlag.png",dpi=140); plt.close(fig)

# ---- 2) spectral structure of contemporaneous return correlation ----
C=np.corrcoef(ret_s.values.T)
evals=np.sort(np.linalg.eigvalsh(C))[::-1]
res["return_corr_eigenvalues"]=[float(x) for x in evals]
res["market_mode_share"]=float(evals[0]/evals.sum())
fig,ax=plt.subplots(figsize=(6,4))
ax.bar(range(1,n+1),evals,color="#1f77b4")
ax.axhline(1,ls=":",color="gray",label="random (eval=1)")
ax.set_xlabel("component");ax.set_ylabel("eigenvalue")
ax.set_title(f"Return correlation spectrum: market mode = {evals[0]/evals.sum()*100:.0f}%")
ax.legend();ax.grid(axis="y",alpha=.3);fig.tight_layout();fig.savefig(f"{OUT}/fig14_spectrum.png",dpi=140);plt.close(fig)

# ---- 3) economic significance with transaction costs ----
half=len(idx)//2
def fit(Xtr,ytr,Xte):
    X1=np.column_stack([np.ones(len(Xtr)),Xtr]); b,*_=np.linalg.lstsq(X1,ytr,rcond=None)
    return np.column_stack([np.ones(len(Xte)),Xte])@b
Xl=ofi_s.values[:-1]
gross_all=[]; net_all=[]; econ={}
for i,ti in enumerate(TICKERS):
    yb=ret[ti].values[1:]  # raw bps next return
    pred=fit(Xl[:half],yb[:half],Xl[half:])
    pos=np.sign(pred)
    rb=yb[half:]
    gross=pos*rb  # bps per bucket
    turn=np.abs(np.diff(np.concatenate([[0],pos])))/2  # position changes
    hs=spr[ti].values[1:][half:]/2  # half-spread bps
    cost=turn*hs
    net=gross-cost
    def ir(x): return x.mean()/x.std()*np.sqrt(len(x)) if x.std()>0 else 0.0
    econ[ti]={"gross_bps_mean":float(gross.mean()),"net_bps_mean":float(net.mean()),
              "gross_IR":float(ir(gross)),"net_IR":float(ir(net)),
              "half_spread_bps":float(np.mean(hs)),"turnover":float(turn.mean())}
    gross_all.append(gross); net_all.append(net)
G=np.mean(np.column_stack(gross_all),axis=1); N=np.mean(np.column_stack(net_all),axis=1)
res["economic"]=econ
res["economic_portfolio"]={"gross_bps_mean":float(G.mean()),"net_bps_mean":float(N.mean()),
    "gross_IR":float(G.mean()/G.std()*np.sqrt(len(G))),"net_IR":float(N.mean()/N.std()*np.sqrt(len(N)))}
fig,ax=plt.subplots(figsize=(6.5,4))
ax.plot(np.cumsum(G),label=f"gross (IR={res['economic_portfolio']['gross_IR']:.2f})",color="#1f77b4")
ax.plot(np.cumsum(N),label=f"net of costs (IR={res['economic_portfolio']['net_IR']:.2f})",color="#d62728")
ax.axhline(0,color="gray",lw=.5)
ax.set_xlabel("test bucket"); ax.set_ylabel("cumulative return (bps)")
ax.set_title("Cross-asset OFI signal: real gross, but costs erase it")
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(f"{OUT}/fig15_economic.png",dpi=140);plt.close(fig)

# ---- 4) sub-sample stability: AAPL own lambda, morning vs afternoon ----
aapl=S["AAPL"].reset_index()
aapl["dmid"]=(aapl["ml"]-aapl["mf"])*100
mornmask=aapl["sod"]<(34200+57600)/2
lam_m=np.polyfit(aapl.loc[mornmask,"o1"],aapl.loc[mornmask,"dmid"],1)[0]
lam_a=np.polyfit(aapl.loc[~mornmask,"o1"],aapl.loc[~mornmask,"dmid"],1)[0]
res["stability_AAPL_lambda"]={"morning":float(lam_m),"afternoon":float(lam_a)}

print(json.dumps(res,indent=2))
