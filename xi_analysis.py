"""
Cross-impact of order flow imbalance across equities
====================================================
Estimates the cross-impact matrix (how asset j's order flow imbalance moves
asset i's price) on real NASDAQ data for five stocks on the same day
(LOBSTER sample: AAPL, AMZN, GOOG, INTC, MSFT, 2012-06-21, level-10 book),
following Cont, Cucuringu & Zhang (2023).

For each asset we build an integrated (PCA across the 10 book levels) OFI and the
mid-price return in common 1-second buckets, standardize both per asset, then:
  * Contemporaneous cross-impact: regress each asset's return on ALL assets' OFI;
    the coefficient matrix is the cross-impact matrix.
  * Compare own-OFI-only R^2 with the full cross-impact R^2 (does cross-asset
    order flow add explanatory power?).
  * Predictive, out of sample: lagged OFI of all assets -> next-bucket return,
    chronological train/test split.

Run:  python3 xi_analysis.py data_dir   (data_dir holds <TICKER>_message_10.csv / _orderbook_10.csv)
"""
from __future__ import annotations
import sys, os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = 10
OUT = "outputs"
BUCKET = 1.0  # seconds


def ofi_buckets(msg_path, book_path):
    msg = pd.read_csv(msg_path, header=None,
                      names=["time", "type", "oid", "size", "price", "dir"])
    book = pd.read_csv(book_path, header=None).values.astype(float)
    t = msg["time"].values

    def lvl(m):
        o = 4 * (m - 1)
        return book[:, o], book[:, o + 1], book[:, o + 2], book[:, o + 3]

    def ofi_level(m):
        ap, asz, bp, bsz = lvl(m)
        dWb = np.where(bp > np.roll(bp, 1), bsz,
              np.where(bp < np.roll(bp, 1), -np.roll(bsz, 1), bsz - np.roll(bsz, 1)))
        dWa = np.where(ap < np.roll(ap, 1), asz,
              np.where(ap > np.roll(ap, 1), -np.roll(asz, 1), asz - np.roll(asz, 1)))
        e = dWb - dWa; e[0] = 0.0
        return e

    E = np.column_stack([ofi_level(m) for m in range(1, LEVELS + 1)])
    mid = (lvl(1)[0] + lvl(1)[2]) / 2 / 1e4
    bk = np.floor(t / BUCKET).astype(int)
    df = pd.DataFrame(E, columns=[f"o{m}" for m in range(1, 11)])
    df["bk"] = bk; df["mid"] = mid
    agg = df.groupby("bk").agg({**{f"o{m}": "sum" for m in range(1, 11)},
                                 "mid": ["first", "last"]})
    agg.columns = [f"o{m}" for m in range(1, 11)] + ["mf", "ml"]
    agg = agg.reset_index()
    agg["ret"] = (agg["ml"] - agg["mf"]) / agg["mf"] * 1e4  # return in bps
    agg = agg.iloc[1:-1]
    # integrated OFI: PC1 across standardized level OFIs
    Z = agg[[f"o{m}" for m in range(1, 11)]].values
    Zs = (Z - Z.mean(0)) / Z.std(0)
    w, V = np.linalg.eigh(np.cov(Zs.T)); pc1 = V[:, -1]
    pc1 = pc1 if pc1.mean() >= 0 else -pc1
    agg["ofi"] = Zs @ pc1
    return agg[["bk", "ofi", "ret"]].set_index("bk")


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta, r2(y, X1 @ beta)


def main():
    os.makedirs(OUT, exist_ok=True)
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "xi_data"
    tickers = []
    for mp in sorted(glob.glob(f"{data_dir}/*_message_10.csv")):
        tickers.append(os.path.basename(mp).split("_")[0])
    tickers = sorted(set(tickers))
    print("tickers:", tickers)

    series = {}
    for tk in tickers:
        mp = glob.glob(f"{data_dir}/{tk}_*_message_10.csv")[0]
        bp = glob.glob(f"{data_dir}/{tk}_*_orderbook_10.csv")[0]
        series[tk] = ofi_buckets(mp, bp)
        print(f"  {tk}: {len(series[tk])} buckets")

    # align on common buckets
    ofi = pd.concat({tk: series[tk]["ofi"] for tk in tickers}, axis=1).dropna()
    ret = pd.concat({tk: series[tk]["ret"] for tk in tickers}, axis=1).dropna()
    common = ofi.index.intersection(ret.index)
    ofi, ret = ofi.loc[common], ret.loc[common]
    # standardize per asset
    ofi = (ofi - ofi.mean()) / ofi.std()
    ret = (ret - ret.mean()) / ret.std()
    print(f"common buckets: {len(common)}")

    n = len(tickers)
    B = np.zeros((n, n))        # cross-impact matrix (contemporaneous)
    r2_cross, r2_own = {}, {}
    X = ofi.values
    for i, tk in enumerate(tickers):
        y = ret[tk].values
        beta, rc = ols(X, y)          # all assets' OFI
        B[i, :] = beta[1:]
        _, ro = ols(ofi[[tk]].values, y)  # own only
        r2_cross[tk] = rc; r2_own[tk] = ro

    # predictive OOS: lagged all-asset OFI -> next-bucket return
    half = len(common) // 2
    pred_cross, pred_own = {}, {}
    Xl = ofi.values[:-1]
    for i, tk in enumerate(tickers):
        yn = ret[tk].values[1:]
        btr, _ = ols(Xl[:half], yn[:half])
        Xte = np.column_stack([np.ones(len(Xl) - half), Xl[half:]])
        pred_cross[tk] = r2(yn[half:], Xte @ btr)
        bo, _ = ols(ofi[[tk]].values[:-1][:half], yn[:half])
        Xteo = np.column_stack([np.ones(len(Xl) - half), ofi[[tk]].values[:-1][half:]])
        pred_own[tk] = r2(yn[half:], Xteo @ bo)

    summary = {
        "tickers": tickers, "common_buckets": int(len(common)),
        "contemporaneous_R2_own": r2_own,
        "contemporaneous_R2_cross": r2_cross,
        "cross_impact_matrix": {tickers[i]: {tickers[j]: float(B[i, j])
                                for j in range(n)} for i in range(n)},
        "predictive_oos_R2_own": pred_own,
        "predictive_oos_R2_cross": pred_cross,
    }
    with open(f"{OUT}/xi_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\nContemporaneous R^2 (own -> cross):")
    for tk in tickers:
        print(f"  {tk}: {r2_own[tk]:.3f} -> {r2_cross[tk]:.3f}  (+{r2_cross[tk]-r2_own[tk]:.3f})")
    print("\nCross-impact matrix (rows=affected asset, cols=source of OFI):")
    print(pd.DataFrame(B, index=tickers, columns=tickers).round(3))

    # --- figure: cross-impact matrix heatmap ---
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(B, cmap="RdBu_r", vmin=-np.abs(B).max(), vmax=np.abs(B).max())
    ax.set_xticks(range(n)); ax.set_xticklabels(tickers)
    ax.set_yticks(range(n)); ax.set_yticklabels(tickers)
    ax.set_xlabel("Source of OFI (asset j)")
    ax.set_ylabel("Affected price (asset i)")
    ax.set_title("Contemporaneous cross-impact matrix")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{B[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(B[i,j]) > np.abs(B).max()*0.5 else "black", fontsize=8)
    fig.colorbar(im, label="impact (standardized)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig9_xi_matrix.png", dpi=140); plt.close(fig)

    # --- figure: own vs cross R^2 ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(n); wdt = 0.38
    ax.bar(x - wdt/2, [r2_own[t] for t in tickers], wdt, label="Own OFI only", color="#bbbbbb")
    ax.bar(x + wdt/2, [r2_cross[t] for t in tickers], wdt, label="All assets' OFI", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("Contemporaneous $R^2$")
    ax.set_title("Does cross-asset OFI add explanatory power?")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_xi_r2.png", dpi=140); plt.close(fig)

    # --- figure: off-diagonal vs diagonal magnitude ---
    diag = np.diag(B); offdiag = B[~np.eye(n, dtype=bool)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([np.abs(diag), np.abs(offdiag)], labels=["own (diagonal)", "cross (off-diagonal)"])
    ax.set_ylabel("|impact coefficient|")
    ax.set_title("Own impact dominates cross impact")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig11_xi_magnitude.png", dpi=140); plt.close(fig)

    print(f"\nWrote figures and xi_summary.json to ./{OUT}/")


if __name__ == "__main__":
    main()
