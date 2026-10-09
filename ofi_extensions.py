"""Extensions: non-linear impact, impact decay (permanent vs transient),
intraday lambda, multi-horizon OOS prediction. Real AAPL LOBSTER data."""
import sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = sys.argv[1]; OUT = "outputs"; LEVELS=10

msg = pd.read_csv(f"{D}/msg.csv", header=None, names=["time","type","oid","size","price","dir"])
book = pd.read_csv(f"{D}/book.csv", header=None).values.astype(float)
t = msg["time"].values

def level(m):
    o=4*(m-1); return book[:,o],book[:,o+1],book[:,o+2],book[:,o+3]
def ofi_level(m):
    ap,asz,bp,bsz=level(m)
    dWb=np.where(bp>np.roll(bp,1),bsz,np.where(bp<np.roll(bp,1),-np.roll(bsz,1),bsz-np.roll(bsz,1)))
    dWa=np.where(ap<np.roll(ap,1),asz,np.where(ap>np.roll(ap,1),-np.roll(asz,1),asz-np.roll(asz,1)))
    e=dWb-dWa; e[0]=0.0; return e
E=np.column_stack([ofi_level(m) for m in range(1,LEVELS+1)])
mid=(level(1)[0]+level(1)[2])/2/1e4

# 1s buckets
bk=np.floor(t).astype(int)
df=pd.DataFrame(E,columns=[f"o{m}" for m in range(1,11)]); df["bk"]=bk; df["mid"]=mid
agg=df.groupby("bk").agg({**{f"o{m}":"sum" for m in range(1,11)},"mid":["first","last"]})
agg.columns=[f"o{m}" for m in range(1,11)]+["mf","ml"]; agg=agg.reset_index()
agg["dmid"]=(agg["ml"]-agg["mf"])*100
# seconds-of-day for the bucket
agg["sod"]=agg["bk"]
agg=agg.iloc[1:-1].reset_index(drop=True)
# integrated OFI
Z=agg[[f"o{m}" for m in range(1,11)]].values; Zs=(Z-Z.mean(0))/Z.std(0)
w,V=np.linalg.eigh(np.cov(Zs.T)); pc1=V[:,-1]; pc1=pc1 if pc1.mean()>=0 else -pc1
agg["oint"]=Zs@pc1
best=agg["o1"].values; y=agg["dmid"].values

res={}

# ---- 1) Non-linear / concave impact curve (binned) ----
q=pd.qcut(agg["o1"],25,duplicates="drop")
curve=agg.groupby(q,observed=True).agg(x=("o1","mean"),yv=("dmid","mean")).reset_index(drop=True)
# fit power law on positive side: y = a*x^beta
pos=curve[(curve.x>0)&(curve.yv>0)]
bx=np.log(pos.x.values); by=np.log(pos.yv.values)
beta,loga=np.polyfit(bx,by,1)
res["impact_exponent_beta"]=float(beta)
# linear vs power R^2 on the full binned curve
def r2(yt,yh): return 1-((yt-yh)**2).sum()/((yt-yt.mean())**2).sum()
lin=np.polyfit(curve.x,curve.yv,1); yhl=np.polyval(lin,curve.x)
pw=np.sign(curve.x)*np.exp(loga)*np.abs(curve.x)**beta
res["curve_R2_linear"]=float(r2(curve.yv.values,yhl))
res["curve_R2_power"]=float(r2(curve.yv.values,pw.values))
fig,ax=plt.subplots(figsize=(6,4.5))
ax.scatter(curve.x,curve.yv,s=18,color="#1f77b4",label="binned means")
xs=np.linspace(curve.x.min(),curve.x.max(),200)
ax.plot(xs,np.sign(xs)*np.exp(loga)*np.abs(xs)**beta,"r-",label=f"power fit ($\\beta$={beta:.2f})")
ax.plot(xs,np.polyval(lin,xs),"g--",alpha=.7,label="linear")
ax.set_xlabel("OFI (1s)");ax.set_ylabel("Mean mid change (cents)")
ax.set_title("Price impact of OFI is approximately linear");ax.legend();ax.grid(alpha=.3)
fig.tight_layout();fig.savefig(f"{OUT}/fig5_linearity.png",dpi=140);plt.close(fig)

# ---- 2) Impact response / permanent vs transient ----
# lambda(l) = cov(cumulative mid change from start t to end t+l, OFI_t)/var(OFI_t)
midend=agg["ml"].values; midstart=agg["mf"].values
o=best; vo=o.var()
L=20; lam=[]
for l in range(0,L+1):
    # cumulative change from start of bucket t to end of bucket t+l
    cum=(np.roll(midend,-l)-midstart)*100
    valid=slice(0,len(o)-l)
    lam.append(np.cov(cum[valid],o[valid])[0,1]/vo)
lam=np.array(lam)
res["impact_l0"]=float(lam[0]); res["impact_l20"]=float(lam[-1])
res["permanent_fraction"]=float(lam[-1]/lam[0])
fig,ax=plt.subplots(figsize=(6,4.2))
ax.plot(range(L+1),lam,"o-",color="#1f77b4")
ax.axhline(lam[0],ls=":",color="gray",label=f"contemporaneous $\\lambda$={lam[0]:.4f}")
ax.set_xlabel("Horizon after OFI (seconds)");ax.set_ylabel("Impact coefficient (cents/unit)")
ax.set_title(f"Impact is permanent: {lam[-1]/lam[0]*100:.0f}% of immediate move persists after {L}s")
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(f"{OUT}/fig6_decay.png",dpi=140);plt.close(fig)

# ---- 3) Intraday variation of lambda (30-min bins) ----
agg["halfhour"]=((agg["sod"]-34200)//1800).astype(int)
rows=[]
for hh,g in agg.groupby("halfhour"):
    if len(g)<50: continue
    b=np.polyfit(g["o1"],g["dmid"],1)
    yh=np.polyval(b,g["o1"]); rows.append((hh,b[0],r2(g["dmid"].values,yh),len(g)))
intr=pd.DataFrame(rows,columns=["hh","lam","r2","n"])
res["intraday_lambda_min"]=float(intr.lam.min()); res["intraday_lambda_max"]=float(intr.lam.max())
clock=[f"{9+ (34200+hh*1800-34200)//3600 +((34200+hh*1800)%3600)//1800*0:.0f}" for hh in intr.hh]
fig,ax=plt.subplots(figsize=(6.5,4))
tlabels=[f"{(34200+h*1800)//3600:02d}:{((34200+h*1800)%3600)//60:02d}" for h in intr.hh]
ax.plot(range(len(intr)),intr.lam,"o-",color="#d62728")
ax.set_xticks(range(len(intr)));ax.set_xticklabels(tlabels,rotation=45,fontsize=7)
ax.set_ylabel("Impact $\\lambda$ (cents/unit)");ax.set_title("Intraday price-impact coefficient (30-min bins)")
ax.grid(alpha=.3);fig.tight_layout();fig.savefig(f"{OUT}/fig7_intraday.png",dpi=140);plt.close(fig)

# ---- 4) Multi-horizon OOS prediction ----
res["pred_oos"]={}
half=len(agg)//2
for h in (1,2,5,10):
    fut=(np.roll(agg["ml"].values,-h)-agg["ml"].values)*100  # cum future return over next h buckets
    d=pd.DataFrame({"x":agg["oint"].values,"y":fut}).iloc[:len(agg)-h].dropna()
    tr=d.iloc[:half]; te=d.iloc[half:]
    b=np.polyfit(tr.x,tr.y,1); yh=np.polyval(b,te.x)
    res["pred_oos"][f"{h}s"]=float(r2(te.y.values,yh))
fig,ax=plt.subplots(figsize=(6,4))
hs=list(res["pred_oos"].keys()); vals=[res["pred_oos"][k] for k in hs]
ax.bar(hs,vals,color="#9467bd")
ax.set_ylabel("Out-of-sample $R^2$");ax.set_xlabel("Prediction horizon")
ax.set_title("Predictive power is ~0 at all horizons (efficiency)")
ax.grid(axis="y",alpha=.3);fig.tight_layout();fig.savefig(f"{OUT}/fig8_pred_horizon.png",dpi=140);plt.close(fig)

print(json.dumps(res,indent=2))
