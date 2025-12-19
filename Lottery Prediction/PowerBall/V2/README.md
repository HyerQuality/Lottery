# Powerball Analysis — Ticket Generation + Backtesting

This repository is a research-oriented framework for:
- generating Powerball tickets using multiple strategies (temperature-mixed and ML-based), and
- replaying those tickets against historical draws to produce a **hypothetical** equity / P&L path.

This project is intended for experimentation and diagnostics, not operational wagering advice.

---

## Repository layout

- `powerball_ticket_generator.py`  
  Temperature-controlled ticket generator (`TemperatureLotteryGenerator`) and a canonical `Ticket` dataclass (used for uniqueness checks across different ticket schemas).

- `powerball_ml_ticket_generator.py`  
  Tree-ensemble ML generator (`PowerballMLTicketGenerator`) that learns per-position categorical distributions from lagged/rolling features, supports hyperparameter tuning, and includes `save_state()` / `load_state()` persistence.

- `powerball_ml_policy.py`  
  Utilities for policy search and an adapter (`MLBacktesterGenerator`) that exposes a `generate_ticket_batch(...)` interface so ML-generated tickets can be plugged into the backtester.

- `powerball_backtester.py`  
  Backtesting engine (`PowerballBacktester`) that replays generated tickets across historical draws, applies a prize table (incl. Power Play multipliers where applicable), and tracks bankroll/equity over time.

- `jackpots_scraper.py`  
  Utility to refresh `jackpots.csv` from the Powerball.net jackpots table.

- `powerball.csv`  
  Historical draw data with columns like: `date`, `white_balls` (pipe-delimited), `red_ball`, `power_play`.

- `jackpots.csv`  
  Jackpot history with columns: `Draw Date`, `Jackpot`, `Winners` (the backtester normalizes column names internally).

---

## Installation

Python 3.10+ recommended.

Core dependencies:
- `numpy`, `pandas`
- `matplotlib`

ML generator dependencies:
- `scikit-learn`
- `joblib`
- `scipy` (for `randint` / `loguniform` parameter distributions)

Optional (performance):
- `numba` (faster payout scoring)

Optional (jackpots scraper):
- `lxml` (used by `pandas.read_html`)

Example:
```bash
pip install numpy pandas matplotlib scikit-learn joblib scipy
pip install numba lxml  # optional
```

---

## Data formats

### `powerball.csv`
Expected columns (aliases are accepted by ML generator; the backtester normalizes common aliases too):
- `date`: draw date (string; parsed by pandas)
- `white_balls`: five white balls as a pipe-delimited string, e.g. `"18|30|40|48|52"`
- `red_ball`: integer 1..26
- `power_play`: e.g. `"2X"`, `"3X"`, `"10X"` (used only when `use_multiplier=True`)

### `jackpots.csv`
Columns:
- `Draw Date`: draw date
- `Jackpot`: string dollars with commas/spaces, e.g. `" 564,100,000 "`
- `Winners`: integer winner count

The backtester normalizes the jackpot file into canonical columns:
- `date`, `jackpot`, `winners`

Jackpot modeling detail:
- If a draw shows `winners > 0`, the backtester models “you as an additional winner” via: `jackpot / (winners + 1)`.
- If the draw date is missing from `jackpots.csv`, the backtester falls back to the median jackpot in the file.

---

## Ticket schema (`Ticket`)

`powerball_ticket_generator.Ticket` is a frozen, canonical representation used for uniqueness checks.

- Whites are stored sorted `(w1 < w2 < w3 < w4 < w5)`.
- Red is stored as `red`.

`Ticket.from_any(...)` accepts:
- tuple: `(w1, w2, w3, w4, w5, red)`
- dict with list whites: `{"white_balls":[...], "red_ball":...}`
- dict “flat” form: `{"white_1":..., ..., "white_5":..., "red_ball":...}`

---

## Generator 1: Temperature-controlled sampling (`TemperatureLotteryGenerator`)

The temperature generator estimates empirical frequencies from `powerball.csv`, then samples from a mixture of:
- empirical probabilities (history-weighted), and
- uniform probabilities.

### Temperature → mixing weight
Let `p_empirical` be the empirical probability vector and `p_uniform` be uniform.

The generator mixes them as:
- `p(T) = (1 - alpha) * p_empirical + alpha * p_uniform`

Where `alpha` is computed in one of two ways:

**Legacy mapping (if `temperature_scale=None`):**
- `alpha = clip(T / max_T, 0, 1)`

**Fixed mapping (recommended; set `temperature_scale`, e.g. 200.0):**
- `alpha = clip(T / temperature_scale, 0, 1)`
- Here, `max_T` acts as a cap on sampled `T` (not a rescaling of alpha).

### Temperature sampling distributions
Controlled by `temperature_sampling`:
- `"uniform"`: uniform over `[T_min, max_T]`
- `"log1p"`: favors lower temperatures (more empirical)
- `"rev_log1p"`: favors higher temperatures (more uniform), with a small tail near low temperatures

### Uniqueness
`generate_ticket_batch(...)` can enforce uniqueness:
- within the returned batch, and
- against an externally provided set via `existing_tickets=...` (used by the backtester to prevent duplicates across ticket pools when multiplier tickets are enabled).

### Example: generate tickets (CSV/cashier-friendly)
```python
import pandas as pd
from powerball_ticket_generator import TemperatureLotteryGenerator

gen = TemperatureLotteryGenerator(
    csv_path="powerball.csv",
    T_white_min=35.0,
    T_red_min=20.0,
    smoothing=1.0,
    temperature_scale=200.0,
    temperature_sampling="rev_log1p",
)

tickets = gen.generate_ticket_batch(
    10,
    max_T=100.0,
    include_metadata=False,  # flat schema for easy CSV printing
    seed=123,
)

df = pd.DataFrame(tickets)
df.to_csv("tickets.csv", index=False)
print(df.head())
```

If `include_metadata=True`, each record includes `T_white`, `T_red`, and `max_T`.

---

## Generator 2: ML tree-ensemble (`PowerballMLTicketGenerator`)

`PowerballMLTicketGenerator` is a time-series supervised learning pipeline that:
- engineers lagged and rolling features from historical draws,
- fits per-position multiclass models for `W1..W5` (69 classes each) and `Red` (26 classes),
- optionally tunes hyperparameters and builds an ensemble,
- generates tickets by sequential sampling without replacement, with a sampling temperature (soften/sharpen).

### Example: fit on a parameter space + evaluate + save state (no embedded draws)

```python
import numpy as np
import pandas as pd

from powerball_ml_ticket_generator import PowerballMLTicketGenerator
from scipy.stats import randint, loguniform

param_space = {
    # With default max_iter (100), LR too small tends to underfit; too large can destabilize.
    "model__learning_rate": loguniform(0.02, 0.12),

    # Keep trees modest; you have ~230 engineered features and (for whites) ~9k augmented rows.
    "model__max_leaf_nodes": randint(15, 80),        # [15..79]

    # Must work for red head too (only ~903 train rows), so don’t push leaf sizes too high.
    "model__min_samples_leaf": randint(10, 80),      # [10..79]

    # Regularization: avoid huge values; log scale gives diversity without wasting samples.
    "model__l2_regularization": loguniform(1e-4, 1.0),

    # Depth: include None for “leaf-node-limited” growth; otherwise keep moderate.
    "model__max_depth": [None, 2, 3, 4, 5, 6],

    # Histogram bin count; trades training speed vs split granularity (higher = finer, slower)
    "model__max_bins": randint(64, 256),   # [64..255]
}

ml = PowerballMLTicketGenerator(
    draw_data="powerball.csv",
    lag_n=15,
    rolling_windows=(5, 10, 20),
    augment_permutations=10,
    use_quantile=False,                 # OFF for boosting
    enable_tuning=True,
    mc_ensemble_size=9,                 # must be >= max ensemble_size you want to test later
    mc_strategy="repeated_random_search",
    tuning_n_iter=15,
    tuning_cv_splits=5,                 # TimeSeriesSplit on train only
    param_distributions=param_space,
    seed=123,
    verbose=True,
).fit()

print(ml.evaluate(split="val"))

# Save without embedding draws (smaller artifact).
ml.save_state("boosted_trees_powerball.joblib", include_draws=False)
```

### Example: load state and generate tickets
If you saved with `include_draws=False`, you must provide draws when loading:

```python
from powerball_ml_ticket_generator import PowerballMLTicketGenerator

ml2 = PowerballMLTicketGenerator.load_state(
    "boosted_trees_powerball.joblib",
    draw_data="powerball.csv",
)

tickets = ml2.generate_tickets(n=20, temperature=1.0, as_flat=True)
print(tickets[0])
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
- “Multiplier” (Power Play) ticket cost: $3

Multiplier application:
- The backtester parses the draw’s `power_play` (e.g. `"3X"`, `"10X"`) into an integer multiplier.
- The multiplier is applied to **non-jackpot prizes only** for tickets that purchased Power Play.
- Jackpot prizes are never multiplied.

### Prize table
The backtester includes explicit constants for the prize tiers it models and applies them via the scoring kernel.

> Note: The project is designed so the prize tiers are defined once (constants) and used consistently by the scoring kernel.

### Example: backtest the temperature generator
```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator(
    csv_path="powerball.csv",
    temperature_scale=200.0,
    temperature_sampling="rev_log1p",
)

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

draw_detail = bt.run()               # returns per-draw pd.DataFrame
ticket_detail = bt.last_ticket_detail
summary = bt.last_summary

bt.plot_winnings(draw_detail)
```

### Example: compare reinvestment policies fairly
`compare_reinvest_rates(...)` returns a MultiIndex DataFrame indexed by `(reinvest_rate, date)`.

Modes:
- `fixed_exposure`: same tickets purchased per draw for all rates (isolates accounting effects)
- `nested_compounding`: reinvestment buys more tickets, but uses shared nested ticket pools per draw so “higher spend includes lower spend tickets” for fairness

```python
cmp = bt.compare_reinvest_rates(
    reinvest_rates=(0.0, 0.25, 0.5, 1.0),
    mode="nested_compounding",
    plot=True,
)
print(cmp.groupby(level=0)[["final_equity", "net_profit_final", "roi"]].tail(1))
```

### Example: analyze temperature stratification
If you ran with `store_temperatures=True`, you can stratify outcomes by temperature deciles:

```python
temp_summary = bt.summarize_by_white_temperature_deciles(bt.last_ticket_detail, q=10)
print(temp_summary)
```

---

## Using ML tickets inside the backtester

The backtester expects a generator that exposes:
```python
generate_ticket_batch(n: int, *, max_T: float, include_metadata: bool, existing_tickets=..., ...)
```

`powerball_ml_policy.MLBacktesterGenerator` adapts a fitted `PowerballMLTicketGenerator` to that interface.

### Example: backtest ML-generated tickets (via adapter)

```python
from powerball_backtester import PowerballBacktester
from powerball_ml_policy import MLBacktesterGenerator
from powerball_ml_ticket_generator import PowerballMLTicketGenerator

ml = PowerballMLTicketGenerator(draw_data="powerball.csv", enable_tuning=False, mc_ensemble_size=7, seed=123).fit()

gen = MLBacktesterGenerator(
    ml,
    temperature=1.0,    # ML sampling temperature
    ensemble_size=5,    # use top-M ensemble members per head
    seed=123,
)

bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen,
    ticket_budget=100,
    use_multiplier=True,
    reinvest_percent=0.0,
    seed=123,
)

draw_detail = bt.run()
bt.plot_winnings(draw_detail)
```

---

## Refreshing `jackpots.csv`
```python
from jackpots_scraper import Jackpots

df = Jackpots(since="2015-01-01").run()
df.to_csv("jackpots.csv", index=False)
```

---

## Known issues / sharp edges

- **Prize tiers vs NumPy fallback:** if `numba` is not installed (so the NumPy scoring kernel is used), verify that all modeled tiers are implemented in the NumPy path. The Numba path uses `_payout_from_counts(...)` (single source of truth). If you notice missing tiers when running without Numba, either install `numba` or update `_score_kernel_numpy(...)` to mirror `_payout_from_counts(...)`.

- **ML adapter uses private helpers:** `MLBacktesterGenerator` calls some private methods on `PowerballMLTicketGenerator` for correct class alignment and white-ball sampling. This is pragmatic but can be refactored into a public API if you want stricter encapsulation.
