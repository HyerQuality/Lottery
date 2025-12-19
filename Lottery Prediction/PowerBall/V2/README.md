# Powerball Analysis — Backtesting + Ticket Generation

This repository is a research-oriented framework for:
- generating Powerball tickets using multiple strategies (temperature-mixed and ML-based), and
- replaying those tickets against historical draws to produce a **hypothetical** equity / P&L path.

This project is intended for experimentation and diagnostics, not operational wagering advice.

---

## Repository layout

- `powerball_ticket_generator.py`  
  Temperature-controlled ticket generator (`TemperatureLotteryGenerator`) + a canonical `Ticket` dataclass.

- `powerball_ml_ticket_generator.py`  
  Tree-ensemble ML model (`PowerballMLTicketGenerator`) that learns per-position (W1..W5, Red) categorical distributions from lagged / rolling features.

- `powerball_backtester.py`  
  Backtesting engine (`PowerballBacktester`) that replays generated tickets across historical draws, applies a simplified prize table, and tracks bankroll/equity over time.

- `jackpots_scraper.py`  
  Utility to refresh `jackpots.csv` from the Powerball.net jackpots table.

- `powerball.csv`  
  Historical draw data with columns: `date`, `white_balls` (pipe-delimited), `red_ball`, `power_play`.

- `jackpots.csv`  
  Jackpot history with columns: `Draw Date`, `Jackpot`, `Winners`.

- `Powerball.ipynb`  
  Notebook scratchpad (exploration / experiments).

---

## Installation

Python 3.10+ recommended.

Core dependencies:
- `numpy`, `pandas`
- `scikit-learn`, `joblib` (ML generator)
- `matplotlib` (plots)
- `python-dateutil` (jackpots scraper + date parsing)

Optional:
- `numba` (faster payout scoring in the backtester)

Example:
```bash
pip install numpy pandas scikit-learn joblib matplotlib python-dateutil numba lxml
```

---

## Data formats

### `powerball.csv`
Columns:
- `date`: draw date (various string formats accepted; the backtester normalizes internally)
- `white_balls`: five white balls as a pipe-delimited string, e.g. `"18|30|40|48|52"`
- `red_ball`: integer 1..26
- `power_play`: e.g. `"2X"`, `"3X"`, `"10X"` (used only when `use_multiplier=True`)

### `jackpots.csv`
Columns:
- `Draw Date`: draw date
- `Jackpot`: string dollars with commas, e.g. `"564,100,000"`
- `Winners`: integer winner count (currently informational; the backtester uses `Jackpot`)

---

## Ticket schema (`Ticket`)

`powerball_ticket_generator.Ticket` is a frozen, canonical representation used for uniqueness checks:
- Whites are stored sorted `(w1 < w2 < w3 < w4 < w5)`.
- Red is stored as `red`.

Conversions via `Ticket.from_any(...)` accept:
- tuple: `(w1, w2, w3, w4, w5, red)`
- dict with list whites: `{"white_balls":[...], "red_ball":...}`
- dict “flat” form: `{"white_1":..., ..., "white_5":..., "red_ball":...}`

---

## Generator 1: Temperature-controlled sampling (`TemperatureLotteryGenerator`)

The temperature generator learns empirical frequencies from `powerball.csv`, then samples using a mixture of:
- empirical probabilities (history-weighted), and
- uniform probabilities.

### Temperature → mixing weight
Let `p_empirical` be the empirical probability vector and `p_uniform` be uniform.

The generator mixes them as:
- `p(T) = (1 - alpha) * p_empirical + alpha * p_uniform`

Where `alpha` is computed in one of two ways:

**Legacy mapping (default):**
- `alpha = clip(T / max_T, 0, 1)`
- `T` is sampled uniformly on `[T_min, max_T]` when `T_white` / `T_red` are not provided.

**Fixed scale mapping (opt-in):**
- Set `temperature_scale` at construction time (e.g. `200.0`)
- `alpha = clip(T / temperature_scale, 0, 1)`
- `max_T` becomes a cap on sampled `T` (not a rescaling of alpha)

### Temperature sampling distributions
Controlled by `temperature_sampling`:
- `"uniform"`: uniform over `[T_min, max_T]`
- `"log1p"`: favors lower temperatures (more empirical)
- `"rev_log1p"`: favors higher temperatures (more uniform), with a small tail near low temperatures

### Uniqueness
`generate_ticket_batch(...)` can enforce uniqueness:
- within the returned batch, and
- against an externally provided set via `existing_tickets=...` (used by the backtester to prevent duplicates across pools).

### Example: generate tickets (CSV/cashier-friendly)
```python
from powerball_ticket_generator import TemperatureLotteryGenerator
import pandas as pd

gen = TemperatureLotteryGenerator(
    csv_path="powerball.csv",
    T_white_min=35.0,
    T_red_min=20.0,
    smoothing=1.0,
    temperature_scale=200.0,          # opt-in (fixed mapping)
    temperature_sampling="rev_log1p", # bias toward “more random”
)

tickets = gen.generate_ticket_batch(
    10,
    max_T=100.0,
    include_metadata=False,  # flat columns for easy CSV printing
    seed=123,
)

df = pd.DataFrame(tickets)
df.to_csv("tickets.csv", index=False)
print(df.head())
```

`include_metadata=True` returns a richer record including `T_white`, `T_red`, and `max_T`.

---

## Generator 2: ML tree-ensemble (`PowerballMLTicketGenerator`)

`PowerballMLTicketGenerator` is a time-series supervised learning pipeline that:
- engineers lagged and rolling features from historical draws,
- fits per-position multiclass models for `W1..W5` (69 classes each) and `Red` (26 classes),
- optionally tunes hyperparameters and builds an ensemble, and
- generates tickets by sequential sampling without replacement.

### Example: fit + generate tickets for the next draw
This is the intended “forecast next draw” workflow.

```python
from powerball_ml_ticket_generator import PowerballMLTicketGenerator

ml = PowerballMLTicketGenerator(
    draw_data="powerball.csv",
    enable_tuning=False,       # faster; set True for randomized search
    mc_ensemble_size=7,
).build().fit()

# Predict probability vectors for the NEXT draw (given the latest available history)
p_whites_by_head, p_red = ml.predict_next_distribution()

# Sample tickets from those distributions
tickets = ml.generate_tickets(n=20, temperature=1.0, as_flat=True)
print(tickets[0])  # e.g. {'white_1':..., ..., 'white_5':..., 'red_ball':...}
```

Notes:
- `temperature` here is an ML sampling *softening/sharpening* control used during generation (independent of the temperature generator above).
- Use `enable_tuning=True` for more rigorous experiments, but it is materially slower.

### Example: evaluate on time splits
The ML class includes `evaluate()` and `print_eval()` helpers that compute log loss and a multiclass Brier score on its internal train/val/test splits.

```python
ml = PowerballMLTicketGenerator("powerball.csv", enable_tuning=False).build().fit()
metrics = ml.evaluate()
ml.print_eval()
```

---

## Backtesting (`PowerballBacktester`)

The backtester replays a generator’s tickets across historical draws and produces per-draw accounting.

### Core accounting model
Per draw:
- You contribute `ticket_budget` dollars (no “ruin” rule).
- Winnings are split:
  - `reinvested = payout * reinvest_percent`
  - `withdrawn  = payout - reinvested`
- `withdrawn` goes into an external account compounding at `withdrawal_apy` (per draw, using `draws_per_year`).
- `reinvested` remains as “bankroll cash” used to buy additional tickets in later draws (when `reinvest_percent > 0`).

Wealth definitions:
- `contributed_t = ticket_budget * t`
- `equity_t = withdrawn_balance_t + bankroll_cash_t`
- `net_profit_t = equity_t - contributed_t`

### Ticket pricing + multiplier modeling
If `use_multiplier=True`, two ticket types are modeled:
- Base ticket cost: $2
- “Multiplier” ticket cost: $3

When multiplier tickets are used, the backtester applies the draw’s `power_play` multiplier to **non-jackpot prizes only** for multiplier tickets (base tickets use multiplier 1×).

### Prize table scope (simplified)
The implemented payout tiers are:

- 5 white + red: **jackpot** (from `jackpots.csv`), not multiplied
- 5 white: $1,000,000 × multiplier
- 4 white + red: $50,000 × multiplier
- 3 white + red: $100 × multiplier
- 2 white + red: $7 × multiplier
- 1 white + red: $4 × multiplier
- 0 white + red: $4 × multiplier

Other official tiers (e.g., 4 white without red) are **not** modeled in this repository snapshot.

### Example: backtest the temperature generator
```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator("powerball.csv", temperature_scale=200.0, temperature_sampling="rev_log1p")

bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen,
    ticket_budget=100,       # dollars contributed each draw
    use_multiplier=True,
    reinvest_percent=0.0,
    max_T=100.0,             # passed through to generate_ticket_batch(...)
    store_temperatures=True, # include T metadata in per-ticket detail
    seed=123,
)

draw_detail = bt.run()              # per-draw P&L table (pandas DataFrame)
ticket_detail = bt.last_ticket_detail
summary = bt.last_summary

bt.plot_winnings(draw_detail)       # 3-panel diagnostics plot
```

### Example: compare reinvest rates
`compare_reinvest_rates(...)` returns a MultiIndex DataFrame indexed by `(reinvest_rate, date)`.

```python
cmp = bt.compare_reinvest_rates(
    reinvest_rates=(0.0, 0.25, 0.5, 1.0),
    mode="nested_compounding",   # or "fixed_exposure"
    plot=True,
)
print(cmp.groupby(level=0)[["final_equity", "net_profit_final", "roi"]].tail(1))
```

### Example: analyze temperature stratification
If you ran with `store_temperatures=True`, you can summarize outcomes by white temperature deciles:

```python
temp_summary = bt.summarize_by_white_temperature_deciles(bt.last_ticket_detail, q=10)
print(temp_summary)
```

---

## Using a custom generator with the backtester

The backtester only requires that `generator` expose:
```python
generate_ticket_batch(n: int, *, max_T: float, include_metadata: bool, ...)
```

If you want to plug in an alternative generator, implement a thin adapter that:
- returns the same ticket dict schema, and
- honors `existing_tickets=` when the backtester requests cross-pool uniqueness.

---

## Refreshing `jackpots.csv`
```python
from jackpots_scraper import Jackpots

df = Jackpots(since="2015-01-01").run()
df.to_csv("jackpots.csv", index=False)
```

---

## Known issues / sharp edges (as of 2025-12-19)

- `powerball_ml_policy.py` currently fails to import due to a missing `PowerballMLTicketGenerator` import (type annotations are evaluated at import time). If you intend to use it, add:
  ```python
  from powerball_ml_ticket_generator import PowerballMLTicketGenerator
  ```
  at the top of the file **or** change the type hints to strings.

- The backtester docstring for `run()` indicates a dict return type, but `run()` returns a pandas DataFrame of draw-level results and stores additional outputs on the instance (`last_ticket_detail`, `last_summary`).

---

## Suggested next enhancements

- Implement the full official prize table, including non-red tiers.
- Add a walk-forward ML adapter that avoids leakage while remaining computationally efficient (caching / incremental refits).
- Add a test suite (e.g., `pytest`) around payout tiers, uniqueness, determinism, and schema invariants.
