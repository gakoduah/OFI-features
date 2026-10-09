"""
Order Flow Imbalance and Price Impact in a Limit Order Book
===========================================================
A from-scratch study of multi-level Order Flow Imbalance (OFI) and its
contemporaneous and predictive relationship with price, on real high-frequency
NASDAQ limit-order-book data (LOBSTER sample: AAPL, 2012-06-21, level-10 book).

Method (after Cont, Kukanov & Stoikov, 2014; multi-level/integrated OFI after
Cont, Cucuringu & Zhang, 2023):
  * Reconstruct best- and deep-level order flow event by event from the book.
  * Aggregate OFI into fixed time buckets at several time scales.
  * Contemporaneous price impact: regress mid-price change on OFI; report R^2
    and the impact coefficient with Newey-West (HAC) standard errors.
  * Compare best-level, multi-level (10), and PCA-integrated OFI.
  * Predictive test: lagged OFI -> next-bucket return, evaluated OUT OF SAMPLE
    on a chronological train/test split (no look-ahead).

Data is NOT redistributed here; see README for the public source.
Run:  python3 ofi_analysis.py data_dir
"""
from __future__ import annotations
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEVELS = 10
OUT = "outputs"


def load(data_dir):
    msg = pd.read_csv(f"{data_dir}/msg.csv", header=None,
                      names=["time", "type", "oid", "size", "price", "dir"])
    book = pd.read_csv(f"{data_dir}/book.csv", header=None).values.astype(float)
    return msg["time"].values, book


def level(book, m):
    o = 4 * (m - 1)
    return book[:, o], book[:, o + 1], book[:, o + 2], book[:, o + 3]  # AskP,AskS,BidP,BidS


def ofi_level(book, m):
    """Per-event order flow imbalance at level m (Cont-Kukanov-Stoikov)."""
    ap, asz, bp, bsz = level(book, m)
    dWb = np.where(bp > np.roll(bp, 1), bsz,
          np.where(bp < np.roll(bp, 1), -np.roll(bsz, 1), bsz - np.roll(bsz, 1)))
    dWa = np.where(ap < np.roll(ap, 1), asz,
          np.where(ap > np.roll(ap, 1), -np.roll(asz, 1), asz - np.roll(asz, 1)))
    e = dWb - dWa
    e[0] = 0.0
    return e


def r2(y, yhat):
    return 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ols(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta, X1, r2(y, X1 @ beta)


def newey_west_se(X1, y, beta, L=10):
    """HAC (Newey-West) standard errors for OLS coefficients."""
    n, k = X1.shape
    resid = y - X1 @ beta
    XtX_inv = np.linalg.inv(X1.T @ X1)
    S = (X1 * resid[:, None]).T @ (X1 * resid[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        Xu = X1 * resid[:, None]
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.diag(cov))


def build_buckets(t, book, E, seconds):
    bucket = np.floor(t / seconds).astype(int)
    df = pd.DataFrame(E, columns=[f"ofi{m}" for m in range(1, LEVELS + 1)])
    df["bucket"] = bucket
    df["mid"] = (level(book, 1)[0] + level(book, 1)[2]) / 2.0 / 1e4
    agg = df.groupby("bucket").agg(
        {**{f"ofi{m}": "sum" for m in range(1, LEVELS + 1)}, "mid": ["first", "last"]})
    agg.columns = [f"ofi{m}" for m in range(1, LEVELS + 1)] + ["mid_first", "mid_last"]
    agg = agg.reset_index()
    agg["dmid"] = (agg["mid_last"] - agg["mid_first"]) * 100.0  # cents
    return agg.iloc[1:-1].reset_index(drop=True)


def integrated(agg):
    Z = agg[[f"ofi{m}" for m in range(1, LEVELS + 1)]].values
    Zs = (Z - Z.mean(0)) / Z.std(0)
    C = np.cov(Zs.T)
    w, V = np.linalg.eigh(C)
    pc1 = V[:, -1]
    if pc1.mean() < 0:
        pc1 = -pc1
    return Zs @ pc1, pc1, w[-1] / w.sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    t, book = load(data_dir)
    E = np.column_stack([ofi_level(book, m) for m in range(1, LEVELS + 1)])
    print(f"events={len(t):,}  span={(t.max()-t.min())/3600:.2f}h")

    summary = {"data": "LOBSTER sample AAPL 2012-06-21 (level-10)",
               "events": int(len(t)), "scales": {}}

    # ---- main analysis at 1s ----
    agg = build_buckets(t, book, E, 1)
    y = agg["dmid"].values
    b1, X1, r2_best = ols(agg[["ofi1"]].values, y)
    se = newey_west_se(X1, y, b1, L=10)
    _, _, r2_ml = ols(agg[[f"ofi{m}" for m in range(1, 11)]].values, y)
    integ, pc1, pc1_var = integrated(agg)
    _, _, r2_int = ols(integ.reshape(-1, 1), y)
    per_level = [ols(agg[[f"ofi{m}"]].values, y)[2] for m in range(1, 11)]

    summary["n_buckets_1s"] = int(len(agg))
    summary["contemporaneous_1s"] = {
        "best_level_R2": r2_best, "multi_level_R2": r2_ml,
        "integrated_R2": r2_int,
        "impact_lambda": b1[1], "lambda_tstat_HAC": b1[1] / se[1],
        "pc1_var_share": pc1_var, "per_level_R2": per_level}
    print(f"[1s] best R2={r2_best:.3f} (lambda t_HAC={b1[1]/se[1]:.1f}), "
          f"multi R2={r2_ml:.3f}, integrated R2={r2_int:.3f}")

    # ---- robustness across time scales ----
    for s in (1, 5, 10, 30):
        a = build_buckets(t, book, E, s)
        yy = a["dmid"].values
        _, _, rb = ols(a[["ofi1"]].values, yy)
        _, _, rm = ols(a[[f"ofi{m}" for m in range(1, 11)]].values, yy)
        ig, _, _ = integrated(a)
        _, _, ri = ols(ig.reshape(-1, 1), yy)
        summary["scales"][f"{s}s"] = {"n": int(len(a)), "best": rb,
                                       "multi": rm, "integrated": ri}
        print(f"[{s}s] n={len(a)} best={rb:.3f} multi={rm:.3f} integrated={ri:.3f}")

    # ---- predictive, out of sample (chronological split) ----
    agg["best"] = agg["ofi1"]
    agg["integ"] = integ
    agg["y_next"] = agg["dmid"].shift(-1)
    pred = agg.dropna().reset_index(drop=True)
    half = len(pred) // 2
    def oos(col):
        tr, te = pred.iloc[:half], pred.iloc[half:]
        b, _, _ = ols(tr[[col]].values, tr["y_next"].values)
        Xte = np.column_stack([np.ones(len(te)), te[col].values])
        return r2(te["y_next"].values, Xte @ b)
    summary["predictive_oos_1s"] = {"best_level": oos("best"),
                                     "integrated": oos("integ")}
    print(f"[pred OOS 1s] best={summary['predictive_oos_1s']['best_level']:.4f} "
          f"integrated={summary['predictive_oos_1s']['integrated']:.4f}")

    # ---- figures ----
    # 1: OFI vs dmid hexbin (best level)
    fig, ax = plt.subplots(figsize=(6, 5))
    m = (np.abs(agg["ofi1"]) < np.percentile(np.abs(agg["ofi1"]), 99))
    ax.hexbin(agg["ofi1"][m], agg["dmid"][m], gridsize=40, cmap="Blues", mincnt=1)
    xs = np.linspace(agg["ofi1"][m].min(), agg["ofi1"][m].max(), 100)
    ax.plot(xs, b1[0] + b1[1] * xs, "r-", lw=1.5, label=f"fit (R$^2$={r2_best:.2f})")
    ax.set_xlabel("Best-level OFI (1s bucket)"); ax.set_ylabel("Mid-price change (cents)")
    ax.set_title("Contemporaneous price impact of OFI"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_impact.png", dpi=140); plt.close(fig)

    # 2: R2 by level
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(1, 11), per_level, color="#1f77b4")
    ax.set_xlabel("Book level"); ax.set_ylabel("Contemporaneous R$^2$")
    ax.set_title("Each level's OFI carries price information")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_level_r2.png", dpi=140); plt.close(fig)

    # 3: R2 vs time scale (best/multi/integrated)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    scales = [1, 5, 10, 30]
    ax.plot(scales, [summary["scales"][f"{s}s"]["best"] for s in scales], "o-", label="Best level")
    ax.plot(scales, [summary["scales"][f"{s}s"]["integrated"] for s in scales], "s-", label="Integrated (PCA)")
    ax.plot(scales, [summary["scales"][f"{s}s"]["multi"] for s in scales], "^-", label="Multi-level (10)")
    ax.set_xlabel("Bucket length (s)"); ax.set_ylabel("Contemporaneous R$^2$")
    ax.set_title("Price-impact fit across time scales"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_scales.png", dpi=140); plt.close(fig)

    # 4: PC1 loadings
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(1, 11), pc1, color="#2ca02c")
    ax.set_xlabel("Book level"); ax.set_ylabel("PC1 loading")
    ax.set_title(f"Integrated-OFI weights (PC1, {pc1_var*100:.0f}% of level-OFI variance)")
    ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_pc1.png", dpi=140); plt.close(fig)

    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote figures and summary.json to ./{OUT}/")


if __name__ == "__main__":
    main()
