# Powerball Temperature-Based Lottery Generator

## Preface

> **AI Co-Author:** ChatGPT (OpenAI) — *GPT-5.2 Thinking*  
> **Role:** Architecture, refactoring strategy, statistical framing, backtesting design review, documentation, and editorial synthesis.
>
> This repository is a study of randomness, but also a record of what happens when you treat an LLM like an engineering partner: design debates, refactors, backtests, failures, and iteration until the code holds.

This project began as a statistical investigation into whether the Powerball lottery—specifically the white balls and the red Powerball—contains any exploitable structure beyond what pure randomness would produce.

Over the course of this analysis, we rigorously evaluated common intuitions and folklore about lottery “patterns” using formal statistical tests, simulations, and information-theoretic tools. The key findings are:

### White Balls (5-of-69)
- Apparent patterns (decade clustering, parity balance, range, smooth cumulative sums) are fully explained by combinatorics and finite-sample randomness.
- Individual white-ball frequencies are statistically uniform within binomial variance.
- No serial dependence, momentum, or regime shifts were detected.
- Change-point tests, conditional tests, and entropy analysis all support a high-entropy, near-ideal random process.
- Any remaining bias, if it exists, is below the threshold required to consistently improve 4/5 or 5/5 hit rates.

### Red Ball (1-of-26)
- The red ball alone behaves as a uniform, memoryless categorical variable.
- Frequency, parity, modulo, recurrence (waiting times), and entropy tests all align with the theoretical ideal.
- A weak conditional signal appeared when conditioning on extreme white-ball ranges, but this effect collapsed under stress testing and was not operationally stable.

### Generator Validation
- We implemented a temperature-controlled generator and formally proved via simulation that:
  - As temperature increases, the generator becomes statistically indistinguishable from uniform random draws.
  - Any introduced bias is fully controlled, bounded, and removable via temperature.
- Ablation studies confirmed that randomizing red-ball temperature adds noise without improving calibration or hit rates.

Bottom line:
There is no hidden exploit in Powerball. What can be done honestly is to control entropy—deciding how much structure vs randomness you want—without pretending that structure implies predictive power.

---

## Project Overview

This repository provides a scientifically defensible Powerball ticket generator built around a single unifying concept:

Temperature controls entropy, not odds.

The generator allows you to interpolate smoothly between:
- Low-entropy, frequency-weighted sampling (exploitative)
- High-entropy, uniform-random sampling (fully random)

All behavior is:
- Statistically validated
- Parameterized
- Reversible
- Explicitly bounded

---

## Core Concepts

### Temperature

Borrowed from statistical mechanics and modern machine learning:

- Low temperature (T → 0): deterministic, sharp distributions
- Moderate temperature (T ≈ 1): structured randomness
- High temperature (T ≥ 5–20): indistinguishable from uniform randomness

Temperature does not change odds—it changes entropy.

### Separate Entropy Channels

- White balls: temperature-controlled, sampled without replacement
- Red ball: fixed-temperature, near-uniform by default

---

## Code Structure

### TemperatureLotteryGenerator

The main class encapsulating all logic.

Key features:
- Empirical frequency extraction from historical data
- Softmax-based temperature scaling
- No-replacement white-ball sampling
- Independent red-ball modeling
- CSV-friendly output mode

---

## Usage Examples

Below are end-to-end examples that use the project’s `.py` modules directly. They are written to be copy/paste runnable from the repository root (where `powerball.csv` and `jackpots.csv` live).

> **Note on “odds” vs “entropy”:** Many examples explore *distributional control* (temperature, sampling mode, policy selection). This changes *how* you randomize, not the underlying lottery odds.

### Example 1 — Refresh `jackpots.csv` by scraping Powerball jackpot history

Use this when you want your `jackpots.csv` aligned to the schema expected by the backtester.

```python
from jackpots_scraper import Jackpots

# Scrape jackpots since 2015 and write to jackpots.csv
Jackpots(since="2015-01-01").to_csv("jackpots.csv")
```

**What this does**
- Scrapes the jackpot table from powerball.net, normalizes column names and formats, and writes a CSV matching this project’s schema:
  - `Draw Date` (m/d/yyyy)
  - `Jackpot` (comma-formatted string with surrounding spaces)
  - `Winners` (int)

**Why it matters**
- `PowerballBacktester` will parse and interpolate jackpot values. If your jackpots file is stale or malformed, backtests can be misleading or fail fast on parsing.

---

### Example 2 — Quick empirical audit: most/least frequent balls in your historical draw file

This is a sanity check: you should see roughly uniform frequencies with binomial noise.

```python
import pandas as pd

df = pd.read_csv("powerball.csv")

# Parse whites from the pipe-delimited string (e.g. "12|33|...")
white = df["white_balls"].astype(str).str.split("|", expand=True).astype(int)
red = df["red_ball"].astype(int)

white_counts = white.stack().value_counts().sort_index()
red_counts = red.value_counts().sort_index()

print("Top 10 white balls by frequency:")
print(white_counts.sort_values(ascending=False).head(10))

print("\nTop 10 red balls by frequency:")
print(red_counts.sort_values(ascending=False).head(10))
```

**What to look for**
- No ball should “run away” from the others beyond what finite-sample randomness allows.
- If you see impossible values (white outside 1–69, red outside 1–26) you likely have a parsing/data issue.

---

### Example 3 — Generate tickets with the temperature generator (metadata mode)

This uses `TemperatureLotteryGenerator`, which mixes empirical frequencies with a uniform distribution based on temperature.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator

gen = TemperatureLotteryGenerator(
    csv_path="powerball.csv",
    temperature_sampling="uniform",   # "uniform" | "log1p" | "rev_log1p"
    temperature_scale=None,           # None => legacy alpha = T/max_T
)

tickets = gen.generate_ticket_batch(
    n=10,
    max_T=100.0,
    include_metadata=True,            # include T_white/T_red in output
)
tickets[:2]
```

**How to interpret the output**
- Each ticket includes:
  - `white_balls`: 5 unique whites sampled without replacement (sorted)
  - `red_ball`: sampled independently
  - `T_white`, `T_red`, `max_T`: the per-ticket temperatures that determined the empirical vs uniform mix

**Typical usage**
- Use `include_metadata=True` when you are doing research and want to stratify or diagnose the effect of temperature on outcomes.

---

### Example 4 — Generate tickets for a cashier / CSV printout (flat schema)

This is the “bring to the counter” mode: flat columns, CSV-friendly.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
import pandas as pd

gen = TemperatureLotteryGenerator("powerball.csv")

tickets = gen.generate_ticket_batch(
    n=50,
    include_metadata=False,  # flat keys: white_1..white_5 + red_ball
)

df = pd.DataFrame(tickets)
df.to_csv("tickets_to_print.csv", index=False)
print(df.head())
```

**Why this format exists**
- It avoids nested lists (`white_balls: [...]`) so you can export cleanly and print/copy the ticket lines easily.

---

### Example 5 — Compare temperature sampling modes (how often you behave “random” vs “frequency-weighted”)

The generator can *sample temperatures* using different shapes. This is about *how often* you operate near T≈0 vs near max_T.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
import numpy as np

def sample_Ts(mode: str, n=5000):
    gen = TemperatureLotteryGenerator("powerball.csv", temperature_sampling=mode)
    batch = gen.generate_ticket_batch(n=n, max_T=100.0, include_metadata=True)
    Tw = np.array([t["T_white"] for t in batch], dtype=float)
    return Tw

for mode in ["uniform", "log1p", "rev_log1p"]:
    Tw = sample_Ts(mode)
    print(mode, "mean(T_white)=", Tw.mean().round(2), "p10/p50/p90=", np.quantile(Tw, [0.1, 0.5, 0.9]).round(2))
```

**Interpretation**
- `uniform`: temperatures spread evenly across the range
- `log1p`: biases toward *lower* T (more empirical structure)
- `rev_log1p`: biases toward *higher* T (mostly uniform randomness, with a small tail of structured samples)

**When to use**
- If you want the generator to be “usually random” but occasionally “structure-seeking,” `rev_log1p` is the sensible choice.

---

### Example 6 — Run a full economic backtest for a fixed ticket budget per draw

This uses the historical draws as the “winning numbers” sequence and simulates your chosen strategy (generator + spending + multiplier + reinvestment).

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator("powerball.csv", temperature_sampling="rev_log1p")

bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen,
    ticket_budget=20,         # dollars per draw you contribute
    use_multiplier=False,     # Power Play tickets cost $3 instead of $2
    reinvest_percent=0.0,     # 0 => no compounding; purely a fixed-budget strategy
    max_T=100.0,
    store_temperatures=True,  # keep T fields in ticket_detail for stratified analysis
    seed=123,
)

draw_detail = bt.run(seed=123)          # per-draw equity & PnL series
ticket_detail = bt.last_ticket_detail   # per-ticket outcomes
summary = bt.last_summary               # quick rollup
summary
```

**What you get**
- `draw_detail`: one row per draw, including spend, payout, equity, net_profit, rolling metrics
- `ticket_detail`: one row per purchased ticket, including the ticket numbers, payout, and (optionally) temperatures

**Why the economics layer matters**
- Many “lottery strategies” ignore the real question: bankroll dynamics. This backtester forces strategy evaluation in dollars, not narratives.

---

### Example 7 — Visualize performance (equity, net profit, cashflows)

```python
# continuing from Example 6
bt.plot_winnings(draw_detail)
```

**What the plot emphasizes**
- Equity vs contributed (with net profit shaded)
- Rolling net profit and rolling volatility/sharpe-style diagnostics
- Spend vs payout per draw with only meaningful win markers to reduce noise

---

### Example 8 — Compare reinvestment policies fairly (two modes)

`compare_reinvest_rates()` lets you compare multiple reinvestment rates. The “nested_compounding” mode avoids unfair randomness artifacts by ensuring higher-spend strategies include the lower-spend ticket set per draw.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator("powerball.csv", temperature_sampling="rev_log1p")

bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen,
    ticket_budget=20,
    use_multiplier=False,
    reinvest_percent=0.0,   # this is overridden inside compare_reinvest_rates()
    seed=7,
)

# Fair comparison with compounding exposure (recommended)
df_nested = bt.compare_reinvest_rates(
    reinvest_rates=(0.0, 0.25, 0.5, 1.0),
    mode="nested_compounding",
    seed=7,
    plot=True,
)

# Accounting-only comparison (same ticket exposure; isolates withdrawal vs reinvest bookkeeping)
df_fixed = bt.compare_reinvest_rates(
    reinvest_rates=(0.0, 0.25, 0.5, 1.0),
    mode="fixed_exposure",
    seed=7,
    plot=False,
)

df_nested.head()
```

**When to use each mode**
- `nested_compounding` (default): use when reinvestment *actually buys more tickets* and you want an apples-to-apples comparison.
- `fixed_exposure`: use when you want to isolate pure accounting effects (withdraw vs keep), holding ticket purchases constant.

---

### Example 9 — Stratify outcomes by temperature deciles (does “more structure” behave differently?)

If you store temperatures, you can bin tickets by their white temperature and compute hit rate / mean payout by bin.

```python
# continuing from Example 6, where store_temperatures=True
temp_summary = bt.summarize_by_white_temperature_deciles(ticket_detail, q=10)
print(temp_summary)
```

**How to read this**
- `hit_rate`: fraction of tickets with payout > 0 in that temperature bin
- `mean_payout`: average payout per ticket in that bin (high variance; don’t over-interpret small samples)
- `n`: count of tickets in the bin

**What you should expect**
- In a fair lottery, stratification should not produce a stable “best bin” that persists out of sample.

---

### Example 10 — Train the ML generator and evaluate calibration metrics

This builds a supervised dataset (predicting draw t+1 from features at draw t), fits an ensemble per head, and reports log loss/Brier scores.

```python
from powerball_ml_ticket_generator import PowerballMLTicketGenerator

ml = PowerballMLTicketGenerator(
    draw_data="powerball.csv",
    lag_n=10,
    rolling_windows=(5, 10, 20),
    enable_tuning=True,        # RandomizedSearchCV per ensemble member
    mc_ensemble_size=7,        # number of candidate models per head
    seed=123,
    verbose=True,
).fit()

ml.print_eval(split="val")
ml.print_eval(split="test")
```

**What this is measuring**
- Log loss and multiclass Brier are *proper scoring rules* for probabilistic classifiers (calibration + sharpness).
- They do not imply profitability. They only indicate whether the model’s probability mass is better than naïve baselines in a predictive sense.

---

### Example 11 — Use the ML generator to produce tickets (next-draw distribution)

```python
# continuing from Example 10
tickets = ml.generate_tickets(n=20, temperature=1.0, as_flat=True)
tickets[:3]
```

**Interpretation**
- The ML generator estimates per-ball probabilities and then samples without replacement for whites (across heads) plus one red sample.
- Temperature here controls *sampling sharpness* from the model’s predicted distributions.

---

### Example 12 — Pick an ML policy on validation, then backtest it end-to-end

This is the project’s most “complete” workflow: choose policy parameters on a validation window, then run an out-of-sample-style backtest using the chosen settings.

```python
from powerball_ml_ticket_generator import PowerballMLTicketGenerator
from powerball_ml_policy import policy_search_on_val, MLBacktesterGenerator
from powerball_backtester import PowerballBacktester

ml = PowerballMLTicketGenerator(draw_data="powerball.csv", seed=123).fit()

best_policy, policy_df = policy_search_on_val(
    ml,
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    ticket_budget=20,
    use_multiplier=False,
)

T_star = float(best_policy["temperature"])
M_star = int(best_policy["ensemble_size"])

gen_star = MLBacktesterGenerator(ml, temperature=T_star, ensemble_size=M_star, seed=123, precompute=True)

bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen_star,
    ticket_budget=20,
    use_multiplier=False,
    reinvest_percent=0.0,
    store_temperatures=True,
    seed=123,
)

draw_detail = bt.run(seed=123)
bt.last_summary
```

**Why this workflow is the right shape**
- It cleanly separates:
  1) model fitting (train/val/test splits inside `PowerballMLTicketGenerator`)
  2) policy selection (grid over temperature + ensemble_size scored on validation window)
  3) economics/backtesting (the same dollar-based machinery as the temperature generator)

---

### Example 13 — Sensitivity analysis: jackpot interpolation assumptions

If your jackpot history file contains only jackpot-winning draws, the backtester must infer “proxy” jackpot values for other draws. You can choose the interpolation policy.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator("powerball.csv")

for policy in ["median", "nearest", "linear", "parabolic"]:
    bt = PowerballBacktester(
        draw_csv="powerball.csv",
        jackpot_csv="jackpots.csv",
        generator=gen,
        ticket_budget=20,
        use_multiplier=False,
        reinvest_percent=0.0,
        jackpot_default=policy,
        seed=123,
    )
    bt.run(seed=123)
    print(policy, bt.last_summary["final_net_profit"])
```

**What this tells you**
- If a result is highly sensitive to `jackpot_default`, you are measuring assumptions about non-winning jackpots rather than a robust property of the generator.

---

### Example 14 — Reproducibility: deterministic runs and generator reseeding

Determinism is critical when comparing strategies; the backtester explicitly supports it.

```python
from powerball_ticket_generator import TemperatureLotteryGenerator
from powerball_backtester import PowerballBacktester

gen = TemperatureLotteryGenerator("powerball.csv")
bt = PowerballBacktester(
    draw_csv="powerball.csv",
    jackpot_csv="jackpots.csv",
    generator=gen,
    ticket_budget=20,
    use_multiplier=False,
    reinvest_percent=0.0,
    seed=999,
)

run_a = bt.run(seed=999)
run_b = bt.run(seed=999)

assert (run_a["net_profit"].values == run_b["net_profit"].values).all()
print("Deterministic: identical net_profit series for same seed.")
```

**Why this matters**
- Without deterministic control, strategy comparisons frequently degenerate into “different random tickets” rather than true policy differences.


## Recommended Defaults

| Parameter | Recommended | Rationale |
|---------|------------|-----------|
| T_white | 1.0 | Best balance of structure vs randomness |
| max_T | 50–100 | Safely saturates uniform regime |
| T_red | 20 | Proven indistinguishable from uniform |
| Multiplier | Optional | Affects payouts only, not odds |

---

## What This Project Is (and Is Not)

### This project is:
- A rigorous statistical exploration
- A transparent entropy-control tool
- A defensible lottery generator
- A demonstration of applied probability, simulation, and inference

### This project is not:
- A system that beats the lottery
- A predictor of winning numbers
- A belief in hidden conspiracies

---

## Final Note

This repository exists to demonstrate how to reason correctly about randomness, bias, and uncertainty—even when the answer is “there is no edge.”

If you’re going to play, play honestly.
If you’re going to model, model rigorously.
