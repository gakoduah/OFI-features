"""Additional inferential statistics for rigor:
 - structural test: impact lambda vs book depth (lambda * D approx const)
 - cross-impact HAC t-stats + Wald test for off-diagonal = 0
 - Clark-West nested OOS test: own vs own+cross predictability
 - integrated-OFI PC1 variance share per asset
"""
import sys, glob, os, json, numpy as np, pandas as pd
D = sys.argv[1]; LEVELS=10; BUCKET=1.0

def load(tk):
    mp=glob.glob(f"{D}/{tk}_*_message_10.csv")[0]; bp=glob.glob(f"{D}/{tk}_*_orderbook_10.csv")[0]
    msg=pd.read_csv(mp,header=None,names=["time","type","oid","size","price","dir"])
    book=pd.read_csv(bp,header=None).values.astype(float); t=msg["time"].values
    def lvl(m): o=4*(m-1); return book[:,o],book[:,o+1],book[:,o+2],book[:,o+3]
    def ofi_level(m):
        ap,asz,bp2,bsz=lvl(m)
        dWb=np.where(bp2>np.roll(bp2,1),bsz,np.where(bp2<np.roll(bp2,1),-np.roll(bsz,1),bsz-np.roll(bsz,1)))
        dWa=np.where(ap<np.roll(ap,1),asz,np.where(ap>np.roll(ap,1),-np.roll(asz,1),asz-np.roll(asz,1)))
        e=dWb-dWa; e[0]=0.0; return e
    E=np.column_stack([ofi_level(m) for m in range(1,11)])
    a1,as1,b1,bs1=lvl(1); mid=(a1+b1)/2/1e4; depth=(as1+bs1)/2.0
    bk=np.floor(t/BUCKET).astype(int)
    df=pd.DataFrame(E,columns=[f"o{m}" for m in range(1,11)]); df["bk"]=bk; df["mid"]=mid; df["dep"]=depth; df["sod"]=t
    g=df.groupby("bk")
    agg=g.agg({**{f"o{m}":"sum" for m in range(1,11)},"mid":["first","last"],"dep":"mean","sod":"first"})
    agg.columns=[f"o{m}" for m in range(1,11)]+["mf","ml","dep","sod"]; agg=agg.reset_index().iloc[1:-1]
    agg["dmid"]=(agg["ml"]-agg["mf"])*100; agg["ret"]=(agg["ml"]-agg["mf"])/agg["mf"]*1e4
    Z=agg[[f"o{m}" for m in range(1,11)]].values; Zs=(Z-Z.mean(0))/Z.std(0)
    w,V=np.linalg.eigh(np.cov(Zs.T)); pc1=V[:,-1]; pc1=pc1 if pc1.mean()>=0 else -pc1
    agg["ofi"]=Zs@pc1
    return agg, float(w[-1]/w.sum())

def nw_cov(X1,resid,L=10):
    n,k=X1.shape; S=(X1*resid[:,None]).T@(X1*resid[:,None])
    for l in range(1,L+1):
        wl=1-l/(L+1); Xu=X1*resid[:,None]; G=Xu[l:].T@Xu[:-l]; S+=wl*(G+G.T)
    XtXi=np.linalg.inv(X1.T@X1); return XtXi@S@XtXi

res={}
# ---- AAPL structural: lambda vs depth ----
aapl,_=load("AAPL")
aapl["hh"]=((aapl["sod"]-34200)//1800).astype(int)
rows=[]
for hh,grp in aapl.groupby("hh"):
    if len(grp)<50: continue
    b=np.polyfit(grp["o1"],grp["dmid"],1)[0]; rows.append((b,grp["dep"].mean()))
L=np.array([r[0] for r in rows]); Dp=np.array([r[1] for r in rows])
res["lambda_depth"]={"corr_lambda_inv_depth":float(np.corrcoef(L,1/Dp)[0,1]),
    "lambda_times_depth_mean":float(np.mean(L*Dp)),"lambda_times_depth_cv":float(np.std(L*Dp)/np.mean(L*Dp)),
    "avg_best_depth":float(aapl["dep"].mean())}
# overall lambda and 1/(2D)
bb=np.polyfit(aapl["o1"],aapl["dmid"],1)[0]
res["lambda_depth"]["overall_lambda_cents"]=float(bb)
res["lambda_depth"]["one_over_2D"]=float(1/(2*aapl["dep"].mean()))

# ---- cross-impact HAC t-stats + Wald test off-diagonal=0 ----
tickers=["AAPL","AMZN","GOOG","INTC","MSFT"]
S={tk:load(tk)[0] for tk in tickers}
pc1share={tk:load(tk)[1] for tk in tickers}
ofi=pd.concat({tk:S[tk].set_index("bk")["ofi"] for tk in tickers},axis=1).dropna()
ret=pd.concat({tk:S[tk].set_index("bk")["ret"] for tk in tickers},axis=1).dropna()
idx=ofi.index.intersection(ret.index); ofi=ofi.loc[idx]; ret=ret.loc[idx]
ofi=(ofi-ofi.mean())/ofi.std(); ret=(ret-ret.mean())/ret.std()
X=ofi.values; n=len(tickers)
tstats={}; wald={}
from numpy.linalg import inv
for i,tk in enumerate(tickers):
    y=ret[tk].values; X1=np.column_stack([np.ones(len(X)),X]); beta,*_=np.linalg.lstsq(X1,y,rcond=None)
    cov=nw_cov(X1,y-X1@beta,L=10); se=np.sqrt(np.diag(cov))
    tstats[tk]={tickers[j]:float(beta[1+j]/se[1+j]) for j in range(n)}
    # Wald test: off-diagonal coefficients (exclude own) jointly zero
    off=[1+j for j in range(n) if j!=i]; bo=beta[off]; Vo=cov[np.ix_(off,off)]
    W=float(bo@inv(Vo)@bo); wald[tk]={"wald_chi2_df4":W}
res["cross_tstats"]=tstats; res["cross_wald_offdiag"]=wald

# ---- Clark-West nested OOS: own vs own+cross ----
half=len(idx)//2
cw={}
for i,tk in enumerate(tickers):
    yn=ret[tk].values[1:]; Xl=X[:-1]
    # small model: own only
    xs=Xl[:,i:i+1]
    def fit_pred(Xtr,ytr,Xte):
        X1=np.column_stack([np.ones(len(Xtr)),Xtr]); b,*_=np.linalg.lstsq(X1,ytr,rcond=None)
        return np.column_stack([np.ones(len(Xte)),Xte])@b
    p1=fit_pred(xs[:half],yn[:half],xs[half:])      # own
    p2=fit_pred(Xl[:half],yn[:half],Xl[half:])      # own+cross (all)
    yte=yn[half:]
    e1=yte-p1; e2=yte-p2
    f=e1**2-(e2**2-(p1-p2)**2)
    cwstat=np.mean(f)/(np.std(f,ddof=1)/np.sqrt(len(f)))
    from math import erfc,sqrt
    cw[tk]={"CW_stat":float(cwstat),"p_value_onesided":float(0.5*erfc(cwstat/sqrt(2)))}
res["clark_west"]=cw
res["pc1_var_share"]=pc1share

print(json.dumps(res,indent=2))

# ---- structural figure (lambda vs 1/depth), reproducible ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as _plt
_lam=[]; _dep=[]
_aapl=aapl
for _hh,_g in _aapl.groupby("hh"):
    if len(_g)<50: continue
    _lam.append(np.polyfit(_g["o1"],_g["dmid"],1)[0]); _dep.append(_g["dep"].mean())
_lam=np.array(_lam); _dep=np.array(_dep)
_fig,_ax=_plt.subplots(figsize=(6,4.5))
_ax.scatter(1/_dep,_lam,s=30,color="#1f77b4")
_c=np.polyfit(1/_dep,_lam,1); _xs=np.linspace((1/_dep).min(),(1/_dep).max(),50)
_ax.plot(_xs,np.polyval(_c,_xs),"r-",label=f"corr={np.corrcoef(_lam,1/_dep)[0,1]:.2f}")
_ax.set_xlabel("1 / average best-level depth (1/shares)"); _ax.set_ylabel("Impact lambda (cents/share)")
_ax.set_title("Price impact scales inversely with depth"); _ax.legend(); _ax.grid(alpha=.3)
import os as _os; _os.makedirs("outputs",exist_ok=True); _fig.tight_layout(); _fig.savefig("outputs/fig12_structural.png",dpi=140)
