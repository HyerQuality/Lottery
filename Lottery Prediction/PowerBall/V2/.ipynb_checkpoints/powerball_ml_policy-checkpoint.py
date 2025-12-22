from __future__ import annotations

"""Powerball ML policy search and backtester generator adapter.

This module provides:

1) policy_search_on_val(...)
   Searches a small policy grid over (temperature, ensemble_size) and evaluates
   each policy by running the PowerballBacktester and scoring only the validation
   draw window.

2) MLBacktesterGenerator
   Adapter that exposes generate_ticket_batch(...) in the shape expected by
   PowerballBacktester, backed by PowerballMLTicketGenerator.

Behavioral compatibility
------------------------
This refactor preserves the external API and default behaviors, with two intended
exceptions:
- Fix iterator seeds exhaustion: `seeds` is materialized once so iterators are
  not consumed across policy evaluations.
- Replace `assert`-based validation with explicit exceptions, so validation is
  enforced even under `python -O`.

The ML generator adapter maintains two execution modes:
- precompute=True (fast): precompute X for all draws once and batch-compute head
  probabilities; per-draw generation is then simple indexing.
- precompute=False (safe): compute the feature row from date-keyed history strictly
  before the current draw date (helps prevent leakage if indices are misaligned).

Notes on indices
----------------
The supervised frame built by PowerballMLTicketGenerator._make_supervised_frame()
has X rows corresponding to historical draws 0..N-2 predicting draws 1..N-1.

Therefore:
- draw index i == 0 has no history: we emit uniform probabilities.
- draw index i > 0 uses X_row == i-1.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import pandas as pd

from powerball_backtester import PowerballBacktester
from powerball_ml_ticket_generator import PowerballMLTicketGenerator


@runtime_checkable
class TicketGenerator(Protocol):
    """Protocol describing the minimal interface expected by PowerballBacktester."""

    def generate_ticket_batch(
        self,
        n: int,
        max_T: float = 100.0,
        include_metadata: bool = True,
        existing_tickets: Optional[Iterable[Dict[str, Any]]] = None,
        ensure_unique: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]: ...


def policy_search_on_val(
    ml: PowerballMLTicketGenerator,
    *,
    draw_csv: str,
    jackpot_csv: str,
    ticket_budget: int,
    temperatures: Sequence[float] = (0.7, 0.9, 1.0, 1.1, 1.3),
    ensemble_sizes: Sequence[int] = (3, 5, 9),
    seeds: Iterable[int] = range(20),
    use_multiplier: bool = False,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Search for the best (temperature, ensemble_size) policy on the ML validation window.

    Parameters
    ----------
    ml:
        Trained PowerballMLTicketGenerator with attributes used for splitting and
        probability generation.
    draw_csv, jackpot_csv:
        Paths passed through to PowerballBacktester.
    ticket_budget:
        Ticket budget per draw.
    temperatures, ensemble_sizes:
        Discrete search grid.
    seeds:
        Random seeds for Monte Carlo evaluation. This iterable is materialized once
        to avoid accidental exhaustion when `seeds` is an iterator.
    use_multiplier:
        Passed through to PowerballBacktester.
    train_frac, val_frac:
        Optional overrides for split fractions when ml.splits_ is None. If omitted,
        defaults are taken from ml.{val_size,test_size}.

    Returns
    -------
    best_policy, policy_df
        best_policy is the top row dict; policy_df contains all evaluated policies.
    """
    seeds_list = list(seeds)  # critical: avoid iterator exhaustion across loops
    if len(seeds_list) == 0:
        raise ValueError("seeds must be a non-empty iterable of ints.")

    n_train, n_val = _infer_train_val_sizes(ml, train_frac=train_frac, val_frac=val_frac)

    # Validation targets correspond to draw indices: (n_train+1) .. (n_train+n_val)
    # Explanation: X row j predicts draw j+1. If training uses X rows [0..n_train-1],
    # that corresponds to target draws [1..n_train]. Validation then targets draws
    # [n_train+1 .. n_train+n_val].
    val_start = n_train + 1
    val_end = n_train + n_val

    results: List[Dict[str, Any]] = []

    for M in ensemble_sizes:
        for T in temperatures:
            vals: List[float] = []
            for s in seeds_list:
                s_int = int(s)

                gen = MLBacktesterGenerator(ml, temperature=T, ensemble_size=M, seed=s_int)
                bt = PowerballBacktester(
                    draw_csv=draw_csv,
                    jackpot_csv=jackpot_csv,
                    generator=gen,
                    ticket_budget=ticket_budget,
                    use_multiplier=use_multiplier,
                    reinvest_percent=0.0,
                    store_temperatures=True,
                    seed=s_int,
                )
                pnl = bt.run(seed=s_int)

                # Score ONLY on validation draw window using incremental profit
                pnl_window = pnl.iloc[val_start : val_end + 1]
                score = float((pnl_window["draw_payout"] - pnl_window["spend"]).sum())
                vals.append(score)

            results.append(
                {
                    "ensemble_size": int(M),
                    "temperature": float(T),
                    "mc_runs": int(len(seeds_list)),
                    "score_mean": float(np.mean(vals)),
                    "score_median": float(np.median(vals)),
                    "score_p10": float(np.quantile(vals, 0.10)),
                    "score_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            )

    df = (
        pd.DataFrame(results)
        .sort_values(["score_median", "score_p10"], ascending=False)
        .reset_index(drop=True)
    )
    best = df.iloc[0].to_dict()
    return best, df


def _infer_train_val_sizes(
    ml: PowerballMLTicketGenerator,
    *,
    train_frac: float | None,
    val_frac: float | None,
) -> Tuple[int, int]:
    """Infer (n_train, n_val) in *X-row space* for validation-window indexing.

    If ml.splits_ exists, use its X_train / X_val lengths.
    Otherwise derive sizes from stored config defaults (val_size/test_size) or
    explicit overrides.
    """
    if getattr(ml, "splits_", None) is not None:
        n_train = len(ml.splits_.X_train)
        n_val = len(ml.splits_.X_val)
        if n_train <= 0 or n_val <= 0:
            raise ValueError(f"Invalid split sizes from ml.splits_: n_train={n_train}, n_val={n_val}")
        return n_train, n_val

    default_val = float(getattr(ml, "val_size", 0.15))
    default_test = float(getattr(ml, "test_size", 0.15))
    default_train = 1.0 - default_val - default_test  # typically 0.70

    val_frac_eff = default_val if val_frac is None else float(val_frac)
    train_frac_eff = default_train if train_frac is None else float(train_frac)
    test_frac = 1.0 - train_frac_eff - val_frac_eff

    if not (0.0 < train_frac_eff < 1.0):
        raise ValueError(f"train_frac must be in (0,1). Got {train_frac_eff}.")
    if not (0.0 < val_frac_eff < 1.0):
        raise ValueError(f"val_frac must be in (0,1). Got {val_frac_eff}.")
    if test_frac < 0.0:
        raise ValueError(
            f"Bad split fracs: train={train_frac_eff}, val={val_frac_eff}, test={test_frac}"
        )

    # X length in _make_supervised_frame is (len(draws) - 1)
    n = len(ml.draws) - 1
    if n <= 1:
        raise ValueError(f"Not enough draws to infer splits. len(ml.draws)={len(ml.draws)}")

    n_test = max(1, int(round(n * test_frac))) if test_frac > 0 else 0
    n_val = max(1, int(round(n * val_frac_eff)))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"train/val/test sizes leave no training data: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    return n_train, n_val


@dataclass(frozen=True)
class _TicketKey:
    """A hashable normalized ticket key used for cross-pool uniqueness."""

    w1: int
    w2: int
    w3: int
    w4: int
    w5: int
    red: int

    @staticmethod
    def from_ticket_dict(t: Dict[str, Any]) -> "_TicketKey":
        """Normalize either schema:
        - {'white_balls': [..], 'red_ball': ..}
        - {'white_1': .., ..., 'white_5': .., 'red_ball': ..}
        """
        if "white_balls" in t:
            whites = list(map(int, t.get("white_balls", [])))
        else:
            whites = [int(t[f"white_{k}"]) for k in range(1, 6)]
        if len(whites) != 5:
            raise ValueError(f"Ticket must have 5 white balls. Got {whites!r}")
        whites.sort()
        red = int(t["red_ball"])
        return _TicketKey(whites[0], whites[1], whites[2], whites[3], whites[4], red)


class MLBacktesterGenerator:
    """Adapter that presents generate_ticket_batch(...) to PowerballBacktester.

    Key behavior:
    - Advances one draw per outer cycle (existing_tickets is None for the first call in a draw).
    - Fast mode (precompute=True): precomputes X for all draws once and batch-computes head probabilities.
    - Safe mode (precompute=False): builds the model feature row from date-keyed history strictly before
      the current draw date (helps detect leakage if indices are misaligned).
    - Enforces uniqueness across multiplier/non-multiplier pools using existing_tickets.
    - Uses only the top-M ensemble members per head (M set at init). If M exceeds
      the available ensemble size, slicing semantics naturally clamp.
    """

    _RED_CHOICES = np.arange(1, 27)

    def __init__(
        self,
        ml: PowerballMLTicketGenerator,
        *,
        temperature: float,
        ensemble_size: int,
        seed: int = 123,
        precompute: bool = True,
    ) -> None:
        self.ml = ml
        self.temperature = float(temperature)
        self.M = int(ensemble_size)
        self.seed = int(seed)

        # Internal state: draw index increments when backtester begins a new draw.
        self._draw_index = -1

        # Stable, 0..N-1 indexed draws (and parsed dates) for robust date-keyed slicing and sanity checks.
        self._draws = self.ml.draws.reset_index(drop=True)
        if "date" not in self._draws.columns:
            raise ValueError("ml.draws must contain a 'date' column.")
        self._draw_dates = pd.to_datetime(self._draws["date"], errors="coerce")

        self._precompute = bool(precompute)
        self._X_all: Optional[pd.DataFrame] = None
        self._pW_all: Optional[Dict[str, np.ndarray]] = None
        self._pR_all: Optional[np.ndarray] = None

        if self._precompute:
            self._build_precompute_caches()

    def reset(self, *, seed: Optional[int] = None) -> None:
        """Reset generator draw state (useful for repeated backtests with the same instance)."""
        self._draw_index = -1
        if seed is not None:
            self.seed = int(seed)

    def _build_precompute_caches(self) -> None:
        """Precompute X and per-head probability arrays for fast per-draw slicing."""
        # X rows correspond to draws 0..N-2 predicting draws 1..N-1
        X_all, _, _ = self.ml._make_supervised_frame(self._draws)
        self._X_all = X_all

        pW_all: Dict[str, np.ndarray] = {}
        for k in range(1, 6):
            head = f"W{k}"
            ens = self.ml.white_ensembles_[head][: self.M]
            pW_all[head] = self.ml.ensemble_proba(
                ens, self._X_all, n_classes=69, class_min=1
            ).astype(np.float32)

        ensR = self.ml.red_ensemble_[: self.M]
        pR_all = self.ml.ensemble_proba(
            ensR, self._X_all, n_classes=26, class_min=1
        ).astype(np.float32)

        self._pW_all = pW_all
        self._pR_all = pR_all

    def _probas_for_draw_index(self, i: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Return (pW_by_head, pR) for draw index i (the draw being scored).

        - i == 0: no history -> uniform
        - i > 0:
            * precompute=True: use X_all row (i-1)
            * precompute=False: compute features from date-keyed history strictly before draw i
        """
        if i <= 0:
            pW = {f"W{k}": np.ones(69, dtype=np.float64) / 69.0 for k in range(1, 6)}
            pR = np.ones(26, dtype=np.float64) / 26.0
            return pW, pR

        if not self._precompute:
            return self._probas_for_draw_index_safe(i)

        return self._probas_for_draw_index_fast(i)

    def _probas_for_draw_index_safe(self, i: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Safe path: build features using date-keyed strict history (slower)."""
        curr_date = self._draw_dates.iloc[i]
        if pd.isna(curr_date):
            raise ValueError(f"Unparseable draw date at i={i}: {self._draws.loc[i, 'date']!r}")

        df_hist = self._draws.loc[self._draw_dates < curr_date].copy()

        if not df_hist.empty:
            hist_max = pd.to_datetime(df_hist["date"], errors="coerce").max()
            if pd.notna(hist_max) and hist_max >= curr_date:
                # Do not use `assert` here: this check should not disappear under -O.
                raise RuntimeError(
                    f"PEEKING DETECTED: hist_max={hist_max} >= current_draw_date={curr_date}"
                )

        X_row = self.ml.features_for_next_draw(df_hist).to_frame().T

        pW: Dict[str, np.ndarray] = {}
        for k in range(1, 6):
            head = f"W{k}"
            ens = self.ml.white_ensembles_[head][: self.M]
            pW[head] = self.ml.ensemble_proba(ens, X_row, n_classes=69, class_min=1)

        ensR = self.ml.red_ensemble_[: self.M]
        pR = self.ml.ensemble_proba(ensR, X_row, n_classes=26, class_min=1)
        return pW, pR

    def _probas_for_draw_index_fast(self, i: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Fast path: use cached batch probabilities from precompute=True."""
        if self._X_all is None or self._pW_all is None or self._pR_all is None:
            raise RuntimeError("precompute=True but caches are missing.")

        row = i - 1
        if row < 0 or row >= len(self._X_all):
            raise IndexError(
                f"Requested row={row} for draw i={i}, but X_all has {len(self._X_all)} rows."
            )

        pW = {f"W{k}": self._pW_all[f"W{k}"][row].astype(np.float64, copy=False) for k in range(1, 6)}
        pR = self._pR_all[row].astype(np.float64, copy=False)
        return pW, pR

    def generate_ticket_batch(
        self,
        n: int,
        max_T: float = 100.0,  # ignored (compat)
        include_metadata: bool = True,
        existing_tickets: Optional[Iterable[Dict[str, Any]]] = None,
        ensure_unique: bool = True,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Generate a batch of tickets for the current draw.

        The PowerballBacktester calls the generator twice per draw when multiplier pools are used:
        - multiplier pool: existing_tickets is None
        - base pool:       existing_tickets is multiplier tickets
        """
        # RNG precedence: provided rng overrides explicit seed overrides self.seed.
        rng = rng or np.random.default_rng(self.seed if seed is None else int(seed))

        # Backtester calls multiplier pool first (existing_tickets=None), then base pool.
        if existing_tickets is None:
            self._draw_index += 1

        i = self._draw_index
        if i < 0:
            raise RuntimeError("Internal draw index went negative; generator state is invalid.")
        if i >= len(self._draws):
            raise IndexError(f"Internal draw index i={i} exceeds available draws (N={len(self._draws)}).")

        pW, pR = self._probas_for_draw_index(i)

        # Uniqueness across pools
        seen: set[_TicketKey] = set()
        if existing_tickets is not None:
            for t in existing_tickets:
                seen.add(_TicketKey.from_ticket_dict(t))

        out: List[Dict[str, Any]] = []
        attempts = 0

        while len(out) < int(n):
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError("Unable to sample enough unique tickets for this draw.")

            whites = self.ml.sample_whites_from_head_probas(
                pW, temperature=self.temperature, rng=rng
            )

            # Apply temperature to red probabilities and sample.
            pR_t = self.ml.apply_temperature(pR.copy(), self.temperature)
            red = int(rng.choice(self._RED_CHOICES, p=pR_t))

            key = _TicketKey(int(whites[0]), int(whites[1]), int(whites[2]), int(whites[3]), int(whites[4]), red)
            if ensure_unique and key in seen:
                continue
            seen.add(key)

            if include_metadata:
                out.append(
                    {
                        "white_balls": whites,
                        "red_ball": red,
                        "white_temperature": float(self.temperature),
                        "red_temperature": float(self.temperature),
                    }
                )
            else:
                out.append({f"white_{j+1}": int(whites[j]) for j in range(5)} | {"red_ball": red})

        return out
