# Order Flow Imbalance and Price Impact in a Limit Order Book

A from-scratch study of **Order Flow Imbalance (OFI)** and its relationship to
price on **real level-10 NASDAQ limit-order-book data**, covering best-level,
multi-level, and PCA-integrated OFI, with both contemporaneous price-impact
estimation and an honest out-of-sample predictive test.

---

## What this does

Given the full event-by-event order book for a trading day, the code reconstructs
order flow imbalance at every book level from first principles and asks:

1. **How much of contemporaneous price movement does OFI explain?** And does the
   *deep* book add to the top of book?
2. **Is the relationship robust** across time scales and under serial-correlation-
   consistent (Newey–West) inference?
3. **Does OFI predict future returns out of sample**, or only move with price?

## Headline results (AAPL, 2012-06-21, 400,391 events)

| Specification | Contemporaneous R² (1s) |
|---|---|
| Best-level OFI | 0.301 |
| Integrated OFI (PCA, 1 factor) | 0.485 |
| Multi-level OFI (10 levels) | 0.505 |

- Explanatory power rises with horizon: multi-level R² = 0.51 (1s) → 0.74 (30s).
- Price-impact coefficient strongly significant under Newey–West (HAC t ≈ 11).
- **Out-of-sample predictive R² ≈ 0.007** — near zero. OFI accompanies price
  formation; it does not forecast it at these horizons. Contemporaneous
  explanation and prediction are deliberately kept separate.

### Deeper analyses (`ofi_extensions.py`)

- **Impact is approximately linear** in OFI (binned R² = 0.94, above a power-law
  fit) — the distinctive property of OFI versus concave trade-size impact.
- **Impact is permanent**: ≈131% of the immediate move persists after 20s (price
  keeps drifting in the OFI direction) — information, not transient pressure.
- **Impact varies intraday**: the coefficient ranges roughly threefold across the
  session, largest near the open.
- **No predictability at any horizon**: out-of-sample R² of 0.006 / 0.005 / 0.003 /
  0.002 at 1 / 2 / 5 / 10 seconds.

### Cross-impact across five stocks (`xi_analysis.py`)

Using all five same-day LOBSTER stocks (AAPL, AMZN, GOOG, INTC, MSFT):

- **Own impact dominates:** cross-impact matrix diagonal 0.64–0.94, every
  off-diagonal below 0.05.
- **Cross-asset OFI adds almost nothing contemporaneously** (R² uplift ≤ 0.009 over
  own-OFI).
- **But it predicts:** lagged cross-asset OFI gives a small, consistent
  out-of-sample R² (~1%) that **beats own-asset OFI** — and for INTC/MSFT, whose own
  flow forecasts nothing, it is the *only* source of predictability. A genuine
  cross-asset lead–lag signal.

Place the five tickers' `message`/`orderbook` CSVs in `data/` and run
`python3 xi_analysis.py data`.

See `report.pdf` for the full write-up (derivations, methodology, figures, and the
embedded source code).

## Method

- **OFI** per event from changes in quoted price/size on each side of the book
  (Cont, Kukanov & Stoikov, 2014), summed into fixed time buckets.
- **Multi-level / integrated OFI** across all ten levels; the integrated factor is
  the first principal component of the standardized level OFIs (after Cont,
  Cucuringu & Zhang, 2023).
- **Price impact**: OLS of mid-price change on OFI with Newey–West standard errors.
- **Prediction**: lagged OFI → next-bucket return, trained on the first half of the
  day and evaluated on the held-out second half (chronological, no look-ahead).

## Data

Real NASDAQ order-book data from the public **LOBSTER** sample (AAPL, 2012-06-21,
level 10). The data is **not redistributed** in this repository; download the
sample from [lobsterdata.com](https://lobsterdata.com) and place the two files in
`data/` as `msg.csv` (message file) and `book.csv` (order-book file).

## Usage

```bash
pip install numpy pandas matplotlib
python3 ofi_analysis.py data
```

Outputs (figures + `summary.json`) are written to `outputs/`.

## Scope

Single asset, single day — enough to establish the OFI–price-impact relationship
rigorously and to separate contemporaneous explanation from out-of-sample
prediction. The multi-asset **cross-impact** extension (how one asset's order flow
moves another's price) needs synchronized multi-ticker data and is the natural next
step.

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book
  events. *Journal of Financial Econometrics*, 12(1), 47–88.
- Cont, R., Cucuringu, M., & Zhang, C. (2023). Cross-impact of order flow imbalance
  in equity markets. *Quantitative Finance*, 23(10), 1373–1393.
- LOBSTER: Limit Order Book System. Sample data, AAPL 2012-06-21.
