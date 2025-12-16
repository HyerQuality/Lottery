# Powerball Analysis — Backtesting Framework

A research-oriented backtesting framework for evaluating **Powerball ticket-generation strategies** against **historical draw data**, with explicit bankroll economics (fixed contributions, optional reinvest/withdraw split, optional withdrawal compounding), optional Power Play-style “multiplier” tickets, and utilities for plotting and temperature-stratified analysis.

The codebase is intentionally small and modular:

- `powerball_ticket_generator.py` — generates candidate tickets (temperature-controlled sampling, uniqueness enforcement, CSV-friendly export).
- `powerball_backtester.py` — replays generated tickets across historical draws, scores payouts, tracks cash/equity over time, and produces diagnostics + plots.
- `powerball.csv` — historical draw data. Sources: https://github.com/jbaranski/jeffs-lottery-utils/blob/main/numbers/powerball.csv | https://catalog.data.gov/dataset/lottery-powerball-winning-numbers-beginning-2010
- `jackpots.csv` — advertised jackpot history + winner counts. Source: https://www.powerball.net/jackpots

---

## Quickstart

### 1) Install dependencies

Minimum dependencies:

- Python 3.10+
- `numpy`
- `pandas`
- `matplotlib`

Optional (recommended for speed):
- `numba` (accelerates the scoring kernel)

Example:

```bash
pip install numpy pandas matplotlib
pip install numba  # optional
```

### 2) Run a baseline backtest

> Note on imports: the provided `powerball_backtester.py` imports the generator as:
> `from V2.powerball_ticket_generator import TemperatureLotteryGenerator`
>
> If your files are **not** inside a `V2/` package folder, either:
> - move both `.py` files into a `V2/` directory with an `__init__.py`, or
> - change that import to `from powerball_ticket_generator import TemperatureLotteryGenerator`.

```python
from V2.powerball_ticket_generator import TemperatureLotteryGenerator
from V2.powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator(csv_path="V2/powerball.csv")

bt = PowerballBacktester(
    draw_csv="V2/powerball.csv",
    jackpot_csv="V2/jackpots.csv",
    generator=gen,
    ticket_budget=100,
    use_multiplier=True,
    reinvest_percent=0.0,     # fixed contribution only (no compounding exposure)
    max_T=50.0,
    store_temperatures=True,  # include per-ticket temperature metadata in ticket_detail
    rolling_window=100,
    prefer_numba=True,        # uses Numba if installed; falls back automatically
    withdrawal_apy=0.0,
)

out = bt.run(seed=123456)

print("Net profit:", out["net_profit"])
print("ROI:", out["roi"])
print(out["draw_detail"].tail())
print(out["ticket_detail"].head())
```

---

## Data files

### `powerball.csv` (draw history)

The backtester normalizes common schema variants, but the canonical expected columns are:

- `date`
- `white_balls` — pipe-delimited list of 5 integers (e.g., `3|14|27|45|62`)
- `red_ball` — integer

### `jackpots.csv` (jackpot history)

Canonical expected columns:

- `date`
- `jackpot` — numeric (commas/spaces tolerated)
- `winners` — integer

#### Jackpot modeling behavior

For each draw date, the backtester computes an effective jackpot value:

- If `winners > 0`, it models **you** as an additional winner and uses:

  `jackpot_value = jackpot / (winners + 1)`

- If the date is missing from `jackpots.csv`, **or** `winners == 0`, it falls back to the **median** jackpot across the provided jackpot history.

This makes “shared jackpot” semantics explicit and avoids divide-by-zero.

---

## Module: `powerball_ticket_generator.py`

### `Ticket` (dataclass)

A frozen, canonical representation of a ticket used for uniqueness checks:

- Whites are stored sorted `(w1 < w2 < w3 < w4 < w5)`.
- Red is stored as `red`.

Supported conversions via `Ticket.from_any(...)`:

- Tuple form: `(w1, w2, w3, w4, w5, red)`
- Dict form with list whites: `{"white_balls":[...], "red_ball":...}`
- Dict “flat” form: `{"white_1":..., ..., "white_5":..., "red_ball":...}`

### `TemperatureLotteryGenerator`

A temperature-controlled generator that learns **empirical ball frequencies** from historical draws and samples new tickets using a temperature-controlled mixture of:

- the empirical distribution (history-weighted frequencies)
- the uniform distribution

#### Probability mixing

For an empirical probability vector `p_empirical` and a temperature `T` bounded by `max_T`, it defines:

- `alpha = clip(T / max_T, 0, 1)`
- `p(T) = (1 - alpha) * p_empirical + alpha * p_uniform`

Interpretation:

- Lower `T` → more history-weighted sampling
- Higher `T` → closer to uniform random sampling

#### Generator parameters

Constructor:

```python
TemperatureLotteryGenerator(
    csv_path: str,
    T_white_min: float = 35.0,
    T_red_min: float = 20.0,
    smoothing: float = 1.0,
)
```

- `csv_path`: used to estimate empirical frequencies.
- `T_white_min`, `T_red_min`: lower bound for per-ticket sampled temperatures when `T_white`/`T_red` are not explicitly provided.
- `smoothing`: Laplace smoothing on counts to avoid zero-probability numbers.

#### `generate_ticket_batch(...)`

```python
generate_ticket_batch(
    n: int,
    max_T: float = 100.0,
    include_metadata: bool = True,
    T_white: Optional[float] = None,
    T_red: Optional[float] = None,
    ensure_unique: bool = True,
    existing_tickets: Optional[Iterable[TicketLike]] = None,
    max_rounds: int = 50,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]
```

Outputs:

- If `include_metadata=True`:

  ```json
  {"white_balls":[w1..w5], "red_ball": r, "T_white":..., "T_red":..., "max_T":...}
  ```

- If `include_metadata=False` (cashier / CSV friendly):

  ```json
  {"white_1":..., "white_2":..., "white_3":..., "white_4":..., "white_5":..., "red_ball":...}
  ```

Uniqueness controls:

- `ensure_unique=True` enforces uniqueness within the returned batch.
- `existing_tickets=...` enforces uniqueness against an external pool as well (critical for “two pools per draw” usage).

Determinism controls:

- Prefer passing an explicit `rng=np.random.default_rng(seed)` for strict reproducibility.
- Alternatively provide `seed=...` and allow the generator to build its own RNG.

---

## Module: `powerball_backtester.py`

### What it does

`PowerballBacktester` orchestrates:

1. Loading and normalizing draw/jackpot CSVs
2. Generating per-draw ticket pools (with optional multiplier tickets)
3. Scoring tickets (NumPy or Numba kernel)
4. Applying bankroll accounting rules:
   - fixed per-draw contributions
   - reinvest vs withdraw split
   - optional compounding of withdrawals
5. Producing:
   - `draw_detail` (one row per draw)
   - `ticket_detail` (one row per ticket purchased)
   - `pnl_table` (compact summary series)
   - scalar rollups (`net_profit`, `roi`, totals, etc.)
6. Plotting utilities and stratified analyses

### Economics model

Each draw:

1. You contribute `ticket_budget` dollars (no “ruin rule”).
2. You have additional **bankroll cash** from prior reinvested winnings (no interest).
3. Available dollars:

   `available = ticket_budget + bankroll_cash`

4. Tickets are purchased using `budget_int = int(available)` dollars (any fractional dollars are ignored).
5. Total payout from tickets for the draw is split:

   - `reinvested = draw_payout * reinvest_percent`
   - `withdrawn  = draw_payout - reinvested`

6. Withdrawals go into an external balance that compounds per draw:

   - per-draw growth: `r_draw = (1 + withdrawal_apy) ** (1 / draws_per_year)`
   - update: `withdrawn_balance = withdrawn_balance * r_draw + withdrawn`

7. Bankroll cash updates as:

   `bankroll_cash = (available - actual_spend) + reinvested`

Wealth definitions:

- `contributed_t = ticket_budget * t`
- `equity_t = withdrawn_balance_t + bankroll_cash_t`
- `net_profit_t = equity_t - contributed_t`

### Ticket pricing and multiplier pools

If `use_multiplier=True`, the backtester models two ticket types:

- Base ticket cost: **$2**
- Multiplier ticket cost: **$3**

Budget allocation rule (`_allocate_ticket_counts`):

- Maximize count of $3 tickets.
- Avoid a remainder of $1 by converting one $3 ticket into two $2 tickets when necessary.

### Uniqueness enforcement across pools (per draw)

When `use_multiplier=True`, the backtester enforces uniqueness across the multiplier and non-multiplier pools by:

- generating multiplier tickets first
- generating non-multiplier tickets with `existing_tickets=<multiplier_batch>`

This requires the generator to support `existing_tickets=...`; the backtester checks that via signature inspection and raises an informative error if missing.

### Prize model implemented

The scoring kernel covers these tiers (USD), with a **2× multiplier** applied to non-jackpot prizes only when `multiplier=True`:

- Jackpot: `5 white + red` → `jackpot_value` (shared if there were existing winners per `jackpots.csv`)
- `5 white (no red)` → `$1,000,000 * m`
- `4 white + red` → `$50,000 * m`
- `3 white + red` → `$100 * m`
- `2 white + red` → `$7 * m`
- `1 white + red` → `$4 * m`
- `0 white + red` → `$4 * m`
- otherwise → `$0`

Where `m = 2.0` if multiplier ticket else `1.0`.

### Determinism and RNG behavior

`run(seed=...)` is designed to be reproducible:

- A local `np.random.Generator` is created via `np.random.default_rng(seed_to_use)`.
- The backtester also uses a temporary “global NumPy seed” context (`_temporary_numpy_seed`) as a best-effort isolation layer (and restores prior RNG state on exit).

### Performance options

- If `prefer_numba=True` and `numba` is installed, scoring uses the Numba-jitted kernel.
- Otherwise it uses a vectorized NumPy kernel.
- If `store_temperatures=False`, the backtester requests tickets without metadata and omits temperature columns from `ticket_detail`.

---

## Public API reference

### Constructor

```python
PowerballBacktester(
    draw_csv: str,
    jackpot_csv: str,
    generator: TemperatureLotteryGenerator,
    ticket_budget: int,
    use_multiplier: bool = True,
    reinvest_percent: float = 0.0,
    max_T: float = 100.0,
    store_temperatures: bool = True,
    rolling_window: int = 30,
    prefer_numba: bool = True,
    withdrawal_apy: float = 0.02,
    draws_per_year: int = 104,
    seed: Optional[int] = None,
)
```

### `run(seed: Optional[int]) -> Dict[str, Any]`

Returns a dictionary containing:

Scalar configuration echoes:
- `seed_used`, `ticket_budget`, `use_multiplier`, `reinvest_percent`, `withdrawal_apy`, `draws_per_year`, `max_T`

DataFrames:
- `draw_detail` — one row per draw, including:
  - `date`, `available`, `spend`, `draw_payout`, `draw_net`
  - `reinvested`, `withdrawn`, `withdrawn_balance`
  - `bankroll_cash`, `equity`, `contributed`, `net_profit`
  - rolling metrics: `rolling_pnl`, `rolling_volatility`, `rolling_sharpe`
- `ticket_detail` — one row per purchased ticket, including:
  - `date`, `cost`, `payout`, `multiplier`
  - `white_1..white_5`, `red_ball`
  - `white_temperature`, `red_temperature` (only if `store_temperatures=True`)
- `pnl_table` — a compact series view of key fields

Rollup metrics:
- `net_profit`, `roi`, `final_equity`, `total_contributed`, `total_spend`, `total_payout`

### `compare_reinvest_rates(...) -> pd.DataFrame`

```python
compare_reinvest_rates(
    reinvest_rates: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
    seed: Optional[int] = None,
    plot: bool = True,
    withdrawal_apy: Optional[float] = None,
    draws_per_year: Optional[int] = None,
    mode: str = "nested_compounding",
) -> pd.DataFrame
```

Modes:

- `fixed_exposure`:
  - Generates tickets based only on `ticket_budget` each draw.
  - All strategies receive the exact same draw payouts; differences reflect only the accounting (withdraw vs retain).
  - Does **not** model “reinvest buys more tickets” by design.

- `nested_compounding`:
  - Models “reinvest buys more tickets”.
  - Ensures fairness by generating shared per-draw ticket pools sized to the maximum demand and giving each strategy a prefix slice of those pools (so higher spend strictly contains the lower spend ticket set).

Returns a MultiIndex DataFrame:
- index: `(reinvest_rate, date)`
- columns: draw-level series and convenience summary columns (e.g., `roi`, `net_profit_final`, `final_equity`).

### `plot_winnings(draw_detail: pd.DataFrame) -> None`

Creates a 3-panel visualization:

1. Equity vs Contributed (net profit shading)
2. Net profit level (left axis) and rolling profit/draw (right axis)
3. Per-draw cashflows (spend vs payout)

Significant-win markers are added only for payouts ≥ $1,000 to reduce clutter.

### `summarize_by_white_temperature_deciles(ticket_detail, q=10) -> pd.DataFrame`

Stratifies ticket outcomes by quantile bins of `white_temperature`:
- `mean_payout`
- `hit_rate` (payout > 0)
- `n` (count)

Requires `store_temperatures=True`.

---

## Running the built-in unit tests

`powerball_backtester.py` includes a lightweight self-test suite (`run_unit_tests`) that validates:

- budget allocation edge cases
- scoring correctness
- NumPy vs Numba kernel parity (if Numba is available)

You can execute it by running the module as a script (assuming paths/imports are configured accordingly):

```bash
python powerball_backtester.py
```

---

## Common usage patterns

### Determinism / regression test

```python
out1 = bt.run(seed=123456)
out2 = bt.run(seed=123456)

assert out1["net_profit"] == out2["net_profit"]
assert out1["draw_detail"].equals(out2["draw_detail"])
assert out1["ticket_detail"].equals(out2["ticket_detail"])
```

### CSV-friendly ticket export (generator only)

```python
tickets = gen.generate_ticket_batch(
    20,
    max_T=50.0,
    include_metadata=False,  # flat schema for cashier/CSV
    ensure_unique=True,
)

import pandas as pd
pd.DataFrame(tickets).to_csv("tickets.csv", index=False)
```

---

## Notes and scope boundaries

- This framework is designed for research and diagnostics, not for operational wagering advice.
- The prize table is intentionally scoped to the tiers implemented in `powerball_backtester.py`.
- The “multiplier” is modeled as a binary 2× factor on non-jackpot prizes; it does not model the full official Power Play multiplier distribution.

---

## Suggested next enhancements

- Implement a full official prize table + variable Power Play multipliers and compare to the simplified model.
- Add an experiment runner (grid over `max_T`, temperature policies, reinvest rates, multiplier on/off) producing a single consolidated report.
- Add a proper test harness (e.g., `pytest`) with fixed fixtures for determinism, uniqueness, and payout tiers.
