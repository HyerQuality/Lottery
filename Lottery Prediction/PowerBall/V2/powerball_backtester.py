import secrets
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple, Sequence, Protocol, runtime_checkable, TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import re

if TYPE_CHECKING:
    from powerball_ticket_generator import TemperatureLotteryGenerator  # noqa: F401

# -------------------------------------------------
# Prize constants (single source of truth)
# -------------------------------------------------
# Official Powerball base prizes for relevant tiers (USD).
# Multiplier (Power Play) is applied to non-jackpot prizes as implemented here.
PRIZE_5_WHITE = 1_000_000.0
PRIZE_4_WHITE_RED = 50_000.0
PRIZE_4_WHITE = 100.0
PRIZE_3_WHITE_RED = 100.0
PRIZE_3_WHITE = 7.0
PRIZE_2_WHITE_RED = 7.0
PRIZE_1_WHITE_RED = 4.0
PRIZE_RED_ONLY = 4.0


# -------------------------------------------------
# NumPy RNG isolation (best-effort: supports Generator rng= when available; falls back to global np.random.* seeding)
# -------------------------------------------------
@contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily set NumPy's *global* RNG seed and restore prior state on exit.

    This is used because TemperatureLotteryGenerator currently relies on np.random.*
    (legacy global RNG). We isolate the seed to keep runs deterministic without polluting
    the caller's global RNG state.
    """
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


# -------------------------------------------------
# Optional Numba JIT (scoring kernel only)
# -------------------------------------------------
try:
    from numba import njit  # type: ignore
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(fn):
            return fn

        return wrap


@njit(cache=True)
def _payout_from_counts(
    white_matches: int,
    red_match: int,
    multiplier_flag: int,  # NOW: actual multiplier for this ticket on this draw (1,2,3,4,5,10,...)
    jackpot_value: float,
) -> float:
    """
    multiplier_flag must be the actual draw multiplier if Power Play was purchased,
    otherwise 1. Jackpot is never multiplied.
    """
    m = float(multiplier_flag)
    if m < 1.0:
        m = 1.0

    # Jackpot: multiplier does not apply.
    if white_matches == 5 and red_match == 1:
        return jackpot_value

    # Non-jackpot tiers.
    if white_matches == 5 and red_match == 0:
        return PRIZE_5_WHITE * m
    if white_matches == 4 and red_match == 1:
        return PRIZE_4_WHITE_RED * m
    if white_matches == 4 and red_match == 0:
        return PRIZE_4_WHITE * m
    if white_matches == 3 and red_match == 1:
        return PRIZE_3_WHITE_RED * m
    if white_matches == 3 and red_match == 0:
        return PRIZE_3_WHITE * m
    if white_matches == 2 and red_match == 1:
        return PRIZE_2_WHITE_RED * m
    if white_matches == 1 and red_match == 1:
        return PRIZE_1_WHITE_RED * m
    if red_match == 1:
        return PRIZE_RED_ONLY * m

    return 0.0


@njit(cache=True)
def _score_kernel_numba(
    ticket_whites: np.ndarray,  # (N, 5) int64
    ticket_reds: np.ndarray,  # (N,) int64
    multipliers: np.ndarray,  # (N,) int16 actual multiplier (1,2,3,4,5,10,...)
    win_whites: np.ndarray,  # (5,) int64
    win_red: int,
    jackpot_value: float,
) -> np.ndarray:
    """Numba-jitted scoring kernel.

    Notes:
      - Uses a loop-based match-count computation for Numba friendliness.
      - Relies on module-level prize constants + _payout_from_counts() for consistency
        with the NumPy fallback.
    """
    n = ticket_whites.shape[0]
    payouts = np.zeros(n, dtype=np.float64)

    for i in range(n):
        wm = 0
        for j in range(5):
            tw = ticket_whites[i, j]
            for k in range(5):
                if tw == win_whites[k]:
                    wm += 1
                    break

        red_match = 1 if ticket_reds[i] == win_red else 0
        payouts[i] = _payout_from_counts(wm, red_match, int(multipliers[i]), float(jackpot_value))

    return payouts


def _score_kernel_numpy(
    ticket_whites: np.ndarray,
    ticket_reds: np.ndarray,
    multipliers: np.ndarray,  # int array: 1 for no PP, else draw PP multiplier
    win_whites: np.ndarray,
    win_red: int,
    jackpot_value: float,
) -> np.ndarray:
    """Vectorized NumPy scoring fallback when Numba is unavailable."""
    white_matches = np.sum(
        ticket_whites[..., None] == win_whites[None, None, :],
        axis=(1, 2),
    )
    red_matches = ticket_reds == win_red

    m = multipliers.astype(np.float64)
    m = np.where(m >= 1.0, m, 1.0)

    payouts = np.zeros(ticket_whites.shape[0], dtype=np.float64)

    is_jackpot = (white_matches == 5) & red_matches
    payouts[is_jackpot] = float(jackpot_value)

    mask = (white_matches == 5) & (~red_matches)
    payouts[mask] = PRIZE_5_WHITE * m[mask]

    mask = (white_matches == 4) & red_matches
    payouts[mask] = PRIZE_4_WHITE_RED * m[mask]

    mask = (white_matches == 4) & (~red_matches)
    payouts[mask] = PRIZE_4_WHITE * m[mask]

    mask = (white_matches == 3) & red_matches
    payouts[mask] = PRIZE_3_WHITE_RED * m[mask]

    mask = (white_matches == 3) & (~red_matches)
    payouts[mask] = PRIZE_3_WHITE * m[mask]

    mask = (white_matches == 2) & red_matches
    payouts[mask] = PRIZE_2_WHITE_RED * m[mask]

    mask = (white_matches == 1) & red_matches
    payouts[mask] = PRIZE_1_WHITE_RED * m[mask]

    mask = (white_matches == 0) & red_matches
    payouts[mask] = PRIZE_RED_ONLY * m[mask]

    return payouts



# -------------------------------------------------
# Public interfaces (typing-only, no runtime coupling)
# -------------------------------------------------
@runtime_checkable
class TicketGenerator(Protocol):
    """Protocol for ticket generators consumed by PowerballBacktester.

    Generators are expected to provide a ``generate_ticket_batch`` method that returns an
    iterable of dict-like ticket records. The backtester adapts to optional keyword
    arguments via signature inspection (rng, seed, ensure_unique, existing_tickets).

    Required keys per ticket when include_metadata=True:
      - white_balls: Sequence[int] (length 5)
      - red_ball: int
    Optional keys (when store_temperatures=True):
      - white_temperature / T_white
      - red_temperature / T_red

    When include_metadata=False (legacy format), required keys:
      - white_1 .. white_5
      - red_ball
    """

    def generate_ticket_batch(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TicketBatch:
    """Internal container for a generated batch of tickets."""

    whites: np.ndarray          # (N, 5) int64
    reds: np.ndarray            # (N,) int64
    multipliers: np.ndarray     # (N,) int16 : 1 if no Power Play; else draw multiplier (2/3/4/5/10)
    costs: np.ndarray           # (N,) int64 : 2 or 3
    white_temps: Optional[np.ndarray]  # (N,) float64 or None
    red_temps: Optional[np.ndarray]    # (N,) float64 or None


class PowerballBacktester:
    """
    Powerball backtesting framework built around TemperatureLotteryGenerator.

    Core economics
    --------------
    • You always contribute `ticket_budget` each draw (no ruin rule).
    • Winnings are split each draw:
        reinvested = payout * reinvest_percent
        withdrawn  = payout - reinvested
    • Withdrawn goes into an external account that compounds at `withdrawal_apy`.
    • Reinvested remains as "bankroll cash" that can be used to buy additional tickets
      in subsequent draws (when reinvest_percent > 0).

    Wealth definitions
    ------------------
    contributed_t = ticket_budget * t
    withdrawn_balance_t compounds per draw at withdrawal_apy
    bankroll_cash_t is the accumulated reinvested cash not yet spent (cash, no return)
    equity_t = withdrawn_balance_t + bankroll_cash_t
    net_profit_t = equity_t - contributed_t

    Seed behavior
    -------------
    • self.seed is set in __init__ via _init_seed(seed).
    • run(seed=None) uses self.seed.
    • run(seed=...) overrides for that run only (does not overwrite self.seed).
    """

    def __init__(
        self,
        draw_csv: str,
        jackpot_csv: str,
        generator: TicketGenerator,
        ticket_budget: int,
        use_multiplier: bool = True,
        reinvest_percent: float = 0.0,
        max_T: float = 100.0,
        store_temperatures: bool = True,
        rolling_window: int = 30,
        prefer_numba: bool = True,
        withdrawal_apy: float = 0.02,
        draws_per_year: int = 104,
        jackpot_default: Literal["median", "nearest", "linear", "parabolic"] = "parabolic",
        seed: Optional[int] = None,
    ):
        self.seed = self._init_seed(seed)

        self.draws = pd.read_csv(draw_csv)
        self.jackpots = pd.read_csv(jackpot_csv)

        self._normalize_date_columns()

        # --- Canonicalize numeric columns ---
        # draws: red_ball should be numeric
        self.draws["red_ball"] = pd.to_numeric(self.draws["red_ball"], errors="coerce").astype("Int64")
        if self.draws["red_ball"].isna().any():
            bad = self.draws[self.draws["red_ball"].isna()].head(5)
            raise ValueError(f"draw_csv has non-numeric red_ball values. Example rows:\n{bad}")

        # jackpots: allow comma/space formatted values (e.g., " 156,200,000 ")
        self.jackpots["jackpot"] = pd.to_numeric(
            self.jackpots["jackpot"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce",
        )
        self.jackpots["winners"] = pd.to_numeric(
            self.jackpots["winners"].astype(str).str.replace(r"[^0-9.-]", "", regex=True),
            errors="coerce",
        ).fillna(0).astype(int)
        if self.jackpots["jackpot"].isna().any():
            bad = self.jackpots[self.jackpots["jackpot"].isna()].head(5)
            raise ValueError(f"jackpot_csv has non-numeric jackpot values. Example rows:\n{bad}")


        if not hasattr(generator, "generate_ticket_batch") or not callable(getattr(generator, "generate_ticket_batch")):
            raise AttributeError(
                "generator must implement a callable generate_ticket_batch(n=..., max_T=..., include_metadata=...)"
            )
        self.generator = generator

        self.ticket_budget = int(ticket_budget)
        self.use_multiplier = bool(use_multiplier)

        # Generator capability check: enforce uniqueness across multiplier/non-multiplier pools per draw.
        sig = inspect.signature(self.generator.generate_ticket_batch)
        params = sig.parameters
        self._generator_supports_existing_tickets = "existing_tickets" in params
        self._generator_supports_rng = "rng" in params
        self._generator_supports_seed = "seed" in params
        self._generator_supports_ensure_unique = "ensure_unique" in params
        if self.use_multiplier and not self._generator_supports_existing_tickets:
            raise AttributeError(
                "Generator must support generate_ticket_batch(..., existing_tickets=...) when use_multiplier=True "
                "to enforce uniqueness across multiplier and non-multiplier ticket pools."
            )


        self.reinvest_percent = float(reinvest_percent)
        if not (0.0 <= self.reinvest_percent <= 1.0):
            raise ValueError("reinvest_percent must be within [0, 1].")

        self.max_T = float(max_T)
        self.store_temperatures = bool(store_temperatures)
        self.rolling_window = int(rolling_window)

        self.withdrawal_apy = float(withdrawal_apy)
        self.draws_per_year = int(draws_per_year)
        if self.draws_per_year <= 0:
            raise ValueError("draws_per_year must be positive.")

        self.jackpot_default = str(jackpot_default).lower()
        if self.jackpot_default not in ("median", "nearest", "linear", "parabolic"):
            raise ValueError("jackpot_default must be one of: median, nearest, linear, parabolic")

        self._use_numba = bool(prefer_numba and NUMBA_AVAILABLE)

        # --- Preprocess draw data for speed ---
        self._draw_dates = self.draws["date"].to_numpy()
        self._win_reds = self.draws["red_ball"].astype(int).to_numpy()

        self._win_whites = np.vstack(
            self.draws["white_balls"]
            .astype(str)
            .str.split("|")
            .apply(lambda xs: [int(x) for x in xs])
            .to_list()
        ).astype(np.int64)

        # --- Jackpot lookup map ---
        self._jackpot_map = dict(
            zip(
                self.jackpots["date"].astype(str),
                zip(self.jackpots["jackpot"], self.jackpots["winners"]),
            )
        )
        self.median_jackpot = float(self.jackpots["jackpot"].median())
        # Precompute anchor points for jackpot interpolation (used when a draw date is absent from jackpot CSV).
        self._prepare_jackpot_anchors()
        self._jackpot_values = np.array(
            [self._jackpot_value_for_date(str(d)) for d in self._draw_dates],
            dtype=np.float64,
        )

        # --- Power Play multiplier per draw (parsed from draws['power_play']) ---
        if "power_play" in self.draws.columns:
            self.draws["power_play_multiplier"] = self.draws["power_play"].apply(self._parse_power_play_value).astype(int)
        else:
            self.draws["power_play_multiplier"] = 1
        
        self._power_play_mult = self.draws["power_play_multiplier"].to_numpy(dtype=np.int16)


        # Stored after run()
        self.pnl_table: Optional[pd.DataFrame] = None

        self._validate_config()

        self._sanity_check()

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        """Validate backtester configuration invariants.

        This method is intentionally conservative: it validates only invariants that should hold
        regardless of generator implementation. It does not mutate state.
        """
        if self.ticket_budget < 0:
            raise ValueError("ticket_budget must be >= 0")

        if not (0.0 <= float(self.reinvest_percent) <= 1.0):
            raise ValueError("reinvest_percent must be within [0, 1].")

        if self.rolling_window <= 0:
            raise ValueError("rolling_window must be positive.")

        if self.draws_per_year <= 0:
            raise ValueError("draws_per_year must be positive.")

        if float(self.withdrawal_apy) < 0.0:
            raise ValueError("withdrawal_apy must be >= 0")

        if self.jackpot_default not in ("median", "nearest", "linear", "parabolic"):
            raise ValueError("jackpot_default must be one of: median, nearest, linear, parabolic")

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------
    def _sanity_check(self) -> None:
        required_methods = [
            "_allocate_ticket_counts",
            "_generate_tickets_for_budget",
            "_score_tickets",
            "_replay_single",
            "run",
            "plot_winnings",
            "compare_reinvest_rates",
        ]
        for name in required_methods:
            if not hasattr(self, name):
                raise RuntimeError(f"Missing required method: {name}")

    # ------------------------------------------------------------------
    # RNG utilities
    # ------------------------------------------------------------------

    def _init_seed(self, seed: Optional[int]) -> int:
        """Normalize / initialize a seed value.

        If `seed` is None, a cryptographically-strong random 32-bit seed is generated.
        If provided, the seed is coerced to an int and mapped into [0, 2**32 - 1).

        Returns the normalized seed.
        """
        if seed is None:
            return int(secrets.randbelow(2**32 - 1))
        s = int(seed)
        # map negatives and large ints into uint32 range deterministically
        return int(s % (2**32 - 1))


    def _normalize_date_columns(self) -> None:
        """Normalize input CSV schemas to expected canonical column names.

        Canonical schema
        ----------------
        Draw CSV:
          - date
          - white_balls (pipe-delimited 5 ints)
          - red_ball

        Jackpot CSV:
          - date
          - jackpot
          - winners

        The normalizer performs a case/whitespace/underscore-insensitive match against a set of
        common aliases (e.g., "Draw Date", "draw_date", "DATE") and renames them to canonical names.
        """
        def _norm(s: str) -> str:
            return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())

        def _rename_to_canonical(df: pd.DataFrame, kind: str, required: Dict[str, Sequence[str]]) -> None:
            # map normalized column name -> original
            norm_to_orig: Dict[str, str] = {_norm(c): c for c in df.columns}

            for canonical, aliases in required.items():
                # search canonical first, then aliases
                candidates = [canonical, *aliases]
                found_orig = None
                for cand in candidates:
                    k = _norm(cand)
                    if k in norm_to_orig:
                        found_orig = norm_to_orig[k]
                        break
                if found_orig is None:
                    raise ValueError(
                        f"{kind} missing required column '{canonical}'. Found: {list(df.columns)}"
                    )
                if found_orig != canonical:
                    df.rename(columns={found_orig: canonical}, inplace=True)

        _rename_to_canonical(
            self.draws,
            "draw_csv",
            required={
                "date": ("draw_date", "drawdate", "draw date", "Date", "DATE"),
                "white_balls": ("whiteballs", "white balls", "whites", "white_numbers", "white numbers"),
                "red_ball": ("redball", "red ball", "powerball", "pb"),
            },
        )

        _rename_to_canonical(
            self.jackpots,
            "jackpot_csv",
            required={
                "date": ("draw_date", "drawdate", "draw date", "Draw Date", "Date", "DATE"),
                "jackpot": ("jackpotamount", "jackpot amount", "Jackpot", "JACKPOT"),
                "winners": ("winnercount", "winner count", "Winners", "WINNERS", "numberofwinners"),
            },
        )

    def reseed(self, seed: Optional[int] = None) -> int:
        """Reseed the instance (updates `self.seed`)."""
        self.seed = self._init_seed(seed)
        return self.seed

    # ------------------------------------------------------------------
    # Jackpot modeling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _date_str_to_ordinal(date_str: str) -> Optional[int]:
        """Parse a date-like string into a Gregorian ordinal (days since 0001-01-01).

        Returns None if parsing fails.
        """
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return None
        # Convert via python date to avoid timezone ambiguity.
        return dt.date().toordinal()

    @staticmethod
    def _quadratic_lagrange(x: float, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Quadratic (parabolic) interpolation using the Lagrange form.

        This is used as a *local* interpolant when jackpot_default='parabolic'. It is parameter-free
        but may overshoot when anchor points vary sharply; results are floored at 0.
        """
        d0 = (x0 - x1) * (x0 - x2)
        d1 = (x1 - x0) * (x1 - x2)
        d2 = (x2 - x0) * (x2 - x1)
        if d0 == 0.0 or d1 == 0.0 or d2 == 0.0:
            # Degenerate (duplicate x values): fall back to linear behavior.
            return float(y1)
        l0 = (x - x1) * (x - x2) / d0
        l1 = (x - x0) * (x - x2) / d1
        l2 = (x - x0) * (x - x1) / d2
        return float(y0) * l0 + float(y1) * l1 + float(y2) * l2

    def _prepare_jackpot_anchors(self) -> None:
        """Prepare sorted jackpot anchor points for interpolation.

        Anchors are taken from jackpot_csv rows (date, jackpot). Duplicate dates keep the last
        occurrence after sorting.
        """
        dt = pd.to_datetime(self.jackpots["date"], errors="coerce")
        df = self.jackpots.loc[~dt.isna(), ["date", "jackpot"]].copy()
        if df.empty:
            self._jackpot_anchor_ord = np.array([], dtype=np.int32)
            self._jackpot_anchor_val = np.array([], dtype=np.float64)
            return

        df["date_ord"] = pd.to_datetime(df["date"], errors="coerce").dt.date.apply(lambda d: d.toordinal())
        df = df.sort_values("date_ord").drop_duplicates("date_ord", keep="last")

        self._jackpot_anchor_ord = df["date_ord"].to_numpy(dtype=np.int32)
        self._jackpot_anchor_val = df["jackpot"].astype(float).to_numpy(dtype=np.float64)

    def _jackpot_default_value(self, draw_date: str) -> float:
        """Return the default jackpot value for a draw date absent from the jackpot map.

        Strategies:
          - 'median':   historical median of anchor jackpots (legacy behavior).
          - 'nearest': nearest anchor jackpot by date.
          - 'linear':  linear interpolation between surrounding anchors.
          - 'parabolic': local quadratic interpolation using 3 anchor points.

        Notes
        -----
        Your jackpot CSV may only contain jackpot-winning draws. In that case, this method provides
        a *proxy* for the jackpot on non-winning draws.
        """
        if self.jackpot_default == "median":
            return float(self.median_jackpot)

        if not hasattr(self, "_jackpot_anchor_ord") or self._jackpot_anchor_ord.size == 0:
            return float(self.median_jackpot)

        x = self._date_str_to_ordinal(draw_date)
        if x is None:
            return float(self.median_jackpot)

        xs = self._jackpot_anchor_ord.astype(np.float64)
        ys = self._jackpot_anchor_val.astype(np.float64)

        if self.jackpot_default == "nearest":
            idx = int(np.argmin(np.abs(xs - float(x))))
            return float(ys[idx])

        if self.jackpot_default == "linear":
            # Linear interpolation is inherently bounded between the two adjacent anchors,
            # but we clamp explicitly for safety (and to align with the parabolic policy below).
            y = float(np.interp(float(x), xs, ys))
            i = int(np.searchsorted(xs, float(x), side="left"))
            if 0 < i < int(xs.shape[0]):
                y_lo = float(min(ys[i - 1], ys[i]))
                y_hi = float(max(ys[i - 1], ys[i]))
                y = min(max(y, y_lo), y_hi)
            return float(max(0.0, y))

        # parabolic (local quadratic)
        n = int(xs.shape[0])
        if n < 3:
            return float(np.interp(float(x), xs, ys))

        # Identify the bracketing segment [k, k+1].
        i = int(np.searchsorted(xs, float(x)))
        if i <= 0:
            return float(ys[0])
        if i >= n:
            return float(ys[-1])

        k = i - 1
        if k <= 0:
            i0, i1, i2 = 0, 1, 2
        elif k >= n - 2:
            i0, i1, i2 = n - 3, n - 2, n - 1
        else:
            i0, i1, i2 = k - 1, k, k + 1

        y = self._quadratic_lagrange(float(x), xs[i0], ys[i0], xs[i1], ys[i1], xs[i2], ys[i2])

        # Clamp to the adjacent anchor jackpots to prevent quadratic overshoot.
        y = float(max(0.0, y))
        y_lo = float(min(ys[k], ys[k + 1]))
        y_hi = float(max(ys[k], ys[k + 1]))
        y = min(max(y, y_lo), y_hi)
        return float(y)

    def _jackpot_value_for_date(self, draw_date: str) -> float:
        """Return the jackpot value applicable to the given draw date.

        The jackpot CSV is expected to provide:
          - jackpot: advertised jackpot amount
          - winners: number of jackpot winners reported for that draw

        Semantics
        ---------
        - If the draw date exists in jackpot_csv and winners > 0, we model *you* as an additional winner:
              payout = jackpot / (winners + 1)
          (Jackpot is not multiplied by Power Play.)
        - If the draw date exists and winners == 0, you are treated as the sole winner:
              payout = jackpot
        - If the date is missing from jackpot_csv, we return a proxy jackpot value using
          the configured ``jackpot_default`` strategy (median/nearest/linear/parabolic).
        """
        row = self._jackpot_map.get(draw_date)
        if row is None:
            return float(self._jackpot_default_value(draw_date))

        jackpot, winners = row
        jackpot_f = float(jackpot)
        try:
            winners_i = int(winners)
        except Exception:
            winners_i = 0

        if winners_i > 0:
            return jackpot_f / (winners_i + 1)

        return jackpot_f

    def _allocate_ticket_counts(self, budget: int) -> Tuple[int, int]:
        """
        Given an integer dollar budget, compute (n_multiplier, n_non_multiplier).
    
        Ticket pricing:
          - $2 base ticket
          - $3 ticket with multiplier
    
        Remainder handling:
          - Avoid remainder==1 by converting one $3 into two $2 tickets.
        """
        budget = int(budget)
        if budget < 2:
            return 0, 0
    
        # If multiplier tickets are disabled, buy only $2 base tickets.
        if not self.use_multiplier:
            return 0, budget // 2
    
        n_mult = budget // 3
        remainder = budget - 3 * n_mult
        if remainder == 1:
            n_mult -= 1
            remainder += 3
    
        n_no = remainder // 2
        return max(0, n_mult), max(0, n_no)


    def _generate_batch(
        self,
        n: int,
        *,
        include_metadata: bool,
        rng: Optional[np.random.Generator],
        existing_tickets: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> Any:
        """Call generator.generate_ticket_batch with the best available deterministic controls."""
        kwargs: Dict[str, Any] = {
            "n": int(n),
            "max_T": float(self.max_T),
            "include_metadata": bool(include_metadata),
        }

        if self._generator_supports_ensure_unique:
            kwargs["ensure_unique"] = True

        if existing_tickets is not None:
            if not self._generator_supports_existing_tickets:
                raise AttributeError("Generator does not support existing_tickets= but uniqueness is required.")
            kwargs["existing_tickets"] = existing_tickets

        # Prefer passing an RNG instance; otherwise fall back to deterministic per-call seeds if supported.
        if rng is not None:
            if getattr(self, "_generator_supports_rng", False):
                kwargs["rng"] = rng
            elif getattr(self, "_generator_supports_seed", False):
                kwargs["seed"] = int(rng.integers(0, 2**32 - 1, dtype=np.uint64))
        else:
            if getattr(self, "_generator_supports_seed", False):
                kwargs["seed"] = int(self.seed)

        return self.generator.generate_ticket_batch(**kwargs)

    def _generate_tickets_for_budget(
        self,
        budget: int,
        *,
        draw_power_play: int,
        rng: Optional[np.random.Generator] = None,
    ) -> TicketBatch:
        """Generate tickets for a single draw given an integer budget.
    
        Returns:
          whites: (N,5) int64
          reds: (N,) int64
          multipliers: (N,) int16  -> 1 for no Power Play, else draw's Power Play (2/3/4/5/10)
          costs: (N,) int64 (2 or 3)
          white_temps: (N,) float64 or None
          red_temps: (N,) float64 or None
        """
        n_mult, n_no = self._allocate_ticket_counts(budget)
        n_total = n_mult + n_no
    
        whites = np.empty((n_total, 5), dtype=np.int64)
        reds = np.empty((n_total,), dtype=np.int64)
        mults = np.empty((n_total,), dtype=np.int16)
        costs = np.empty((n_total,), dtype=np.int64)
    
        include_metadata = bool(self.store_temperatures)
        white_temps = np.empty((n_total,), dtype=np.float64) if include_metadata else None
        red_temps = np.empty((n_total,), dtype=np.float64) if include_metadata else None
    
        idx = 0
        draw_pp = int(draw_power_play)
        if draw_pp < 1:
            draw_pp = 1
    
        def _fill(batch: Iterable[Dict[str, Any]], is_mult: bool, cost: int) -> None:
            nonlocal idx
            for t in batch:
                if include_metadata:
                    w = t["white_balls"]
                    r = t["red_ball"]
                    if white_temps is not None:
                        white_temps[idx] = float(t.get("white_temperature", t.get("T_white", np.nan)))
                    if red_temps is not None:
                        red_temps[idx] = float(t.get("red_temperature", t.get("T_red", np.nan)))
                else:
                    w = (t["white_1"], t["white_2"], t["white_3"], t["white_4"], t["white_5"])
                    r = t["red_ball"]
    
                whites[idx, :] = (int(w[0]), int(w[1]), int(w[2]), int(w[3]), int(w[4]))
                reds[idx] = int(r)
    
                # KEY CHANGE:
                # - If Power Play ticket purchased: multiplier is draw_pp (2/3/4/5/10)
                # - Else: multiplier is 1
                mults[idx] = draw_pp if is_mult else 1
    
                costs[idx] = int(cost)
                idx += 1
    
        if n_mult > 0:
            batch_mult = self._generate_batch(
                n_mult,
                include_metadata=include_metadata,
                rng=rng,
            )
            _fill(batch_mult, is_mult=True, cost=3)
        else:
            batch_mult = None
    
        if n_no > 0:
            batch_no = self._generate_batch(
                n_no,
                include_metadata=include_metadata,
                rng=rng,
                existing_tickets=batch_mult,
            )
            _fill(batch_no, is_mult=False, cost=2)
    
        return TicketBatch(
            whites=whites,
            reds=reds,
            multipliers=mults,
            costs=costs,
            white_temps=white_temps,
            red_temps=red_temps,
        )

    def _score_tickets(
        self,
        whites: np.ndarray,
        reds: np.ndarray,
        mults: np.ndarray,
        win_whites: np.ndarray,
        win_red: int,
        jackpot_value: float,
    ) -> np.ndarray:
        if whites.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        if self._use_numba:
            return _score_kernel_numba(
                whites,
                reds,
                mults.astype(np.int16),
                win_whites.astype(np.int64),
                int(win_red),
                float(jackpot_value),
            )

        return _score_kernel_numpy(
            whites,
            reds,
            mults,
            win_whites.astype(np.int64),
            int(win_red),
            float(jackpot_value),
        )

    @staticmethod
    def _parse_power_play_value(v: Any) -> int:
        """
        Parse draw-level Power Play strings like '3X', '10x', ' 4 X ' into an integer multiplier.
        Returns 1 if missing/unparseable.
        """
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 1
        s = str(v).strip().upper()
        if s == "" or s in {"NAN", "NONE"}:
            return 1
        m = re.search(r"(\d+)", s)
        if not m:
            return 1
        x = int(m.group(1))
        return x if x >= 1 else 1

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------
    def _add_rolling_metrics(self, draw_df: pd.DataFrame) -> pd.DataFrame:
        df = draw_df.copy()
        w = int(self.rolling_window)
        minp = max(2, w // 3)

        df["rolling_pnl"] = df["draw_net"].rolling(window=w, min_periods=minp).sum()
        df["rolling_volatility"] = df["draw_net"].rolling(window=w, min_periods=minp).std(ddof=1)

        roll_mean = df["draw_net"].rolling(window=w, min_periods=minp).mean()
        roll_std = df["rolling_volatility"]

        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = (roll_mean / roll_std) * np.sqrt(float(w))

        df["rolling_sharpe"] = sharpe.where(roll_std > 0)
        return df

    def _compute_withdrawal_growth(self, dates: pd.Series) -> np.ndarray:
        """
        Per-draw compounding multipliers for the withdrawal account using actual time deltas.
        Returns growth array with growth[0] = 1.0, and for i>0:
            growth[i] = (1 + withdrawal_apy) ** delta_years[i]
        Falls back to draws_per_year if dates are missing/unparseable.
        """
        dates = pd.to_datetime(dates, errors="coerce")
        n = int(len(dates))
        growth = np.ones(n, dtype=np.float64)
        if n <= 1:
            return growth
    
        if dates.isna().any():
            # fallback: constant cadence
            r_draw = (1.0 + float(self.withdrawal_apy)) ** (1.0 / float(self.draws_per_year))
            growth[:] = r_draw
            growth[0] = 1.0
            return growth
    
        delta_days = np.diff(dates.values).astype("timedelta64[D]").astype(np.float64)
        delta_years = np.maximum(delta_days / 365.25, 0.0)
        growth[1:] = (1.0 + float(self.withdrawal_apy)) ** delta_years
        return growth

    def _replay_single(self, *, rng: Optional[np.random.Generator]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
        """
        Single-run replay using the configured reinvest_percent.

        This is the compounding exposure model:
          - Each draw contributes ticket_budget
          - bankroll_cash accumulates reinvested winnings (cash, no interest)
          - available dollars each draw = ticket_budget + bankroll_cash
          - buy as many tickets as possible with floor(available)
          - carry any unspent remainder in bankroll_cash
        """
        # Grow the withdrawal account by the *actual* time delta between draws.
        # This avoids assuming a fixed draws_per_year (Powerball cadence changed over time).
        dates = pd.to_datetime(self._draw_dates)

        growth = np.ones(len(self.draws), dtype=np.float64)
        try:
            if len(dates) > 1 and not pd.isna(dates).any():
                delta_days = np.diff(dates).astype("timedelta64[D]").astype(np.float64)
                delta_years = np.maximum(delta_days / 365.25, 0.0)
                growth[1:] = (1.0 + float(self.withdrawal_apy)) ** delta_years
            else:
                # Fallback: assume constant draw cadence
                r_draw = (1.0 + float(self.withdrawal_apy)) ** (1.0 / float(self.draws_per_year))
                growth[:] = r_draw
                growth[0] = 1.0
        except Exception:
            # Very defensive fallback
            r_draw = (1.0 + float(self.withdrawal_apy)) ** (1.0 / float(self.draws_per_year))
            growth[:] = r_draw
            growth[0] = 1.0


        withdrawn_balance = 0.0
        bankroll_cash = 0.0
        cumulative_contributed = 0.0

        draw_records = []
        ticket_frames = []

        for i in range(len(self.draws)):
            cumulative_contributed += self.ticket_budget

            available = float(self.ticket_budget) + float(bankroll_cash)
            budget_int = int(available)

            draw_pp = int(self._power_play_mult[i])
            batch = self._generate_tickets_for_budget(
                budget_int,
                rng=rng,
                draw_power_play=draw_pp,
            )
            actual_spend = float(batch.costs.sum()) if batch.costs.size > 0 else 0.0

            payouts = self._score_tickets(
                whites=batch.whites,
                reds=batch.reds,
                mults=batch.multipliers,
                win_whites=self._win_whites[i],
                win_red=int(self._win_reds[i]),
                jackpot_value=float(self._jackpot_values[i]),
            )
            draw_payout = float(payouts.sum())

            reinvested = float(draw_payout * self.reinvest_percent)
            withdrawn = float(draw_payout - reinvested)

            withdrawn_balance = withdrawn_balance * float(growth[i]) + withdrawn

            # Update bankroll cash:
            # - leftover from (available - actual_spend)
            # - plus reinvested winnings
            bankroll_cash = (available - actual_spend) + reinvested

            equity = withdrawn_balance + bankroll_cash
            net_profit = equity - cumulative_contributed
            draw_net = draw_payout - actual_spend

            if batch.whites.shape[0] > 0:
                td = {
                    "date": np.array([self._draw_dates[i]] * batch.whites.shape[0], dtype=object),
                    "cost": batch.costs.astype(np.int64),
                    "payout": payouts.astype(np.float64),
                    "multiplier": batch.multipliers.astype(np.int16),
                    "white_1": batch.whites[:, 0].astype(np.int64),
                    "white_2": batch.whites[:, 1].astype(np.int64),
                    "white_3": batch.whites[:, 2].astype(np.int64),
                    "white_4": batch.whites[:, 3].astype(np.int64),
                    "white_5": batch.whites[:, 4].astype(np.int64),
                    "red_ball": batch.reds.astype(np.int64),
                }
                if self.store_temperatures:
                    td["white_temperature"] = batch.white_temps.astype(np.float64)  # type: ignore[union-attr]
                    td["red_temperature"] = batch.red_temps.astype(np.float64)  # type: ignore[union-attr]
                ticket_frames.append(pd.DataFrame(td))

            draw_records.append({
                "date": self._draw_dates[i],
                "available": float(available),
                "spend": float(actual_spend),
                "draw_payout": float(draw_payout),
                "draw_net": float(draw_net),
                "reinvested": float(reinvested),
                "withdrawn": float(withdrawn),
                "withdrawn_balance": float(withdrawn_balance),
                "bankroll_cash": float(bankroll_cash),
                "equity": float(equity),
                "contributed": float(cumulative_contributed),
                "net_profit": float(net_profit),
            })

        draw_df = pd.DataFrame(draw_records)
        draw_df = self._add_rolling_metrics(draw_df)
        ticket_df = pd.concat(ticket_frames, ignore_index=True) if ticket_frames else pd.DataFrame()

        summary = {
            "final_equity": float(draw_df["equity"].iloc[-1]) if not draw_df.empty else 0.0,
            "final_net_profit": float(draw_df["net_profit"].iloc[-1]) if not draw_df.empty else 0.0,
            "total_contributed": float(draw_df["contributed"].iloc[-1]) if not draw_df.empty else 0.0,
            "total_payout": float(draw_df["draw_payout"].sum()) if not draw_df.empty else 0.0,
            "total_spend": float(draw_df["spend"].sum()) if not draw_df.empty else 0.0,
        }
        summary["roi"] = (summary["final_equity"] / summary["total_contributed"]) if summary["total_contributed"] > 0 else np.nan

        return draw_df, ticket_df, summary

    # ------------------------------------------------------------------
    # compare_reinvest_rates() engines
    # ------------------------------------------------------------------
    def _comparison_outputs_to_df(self, outputs: Dict[float, Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert compare outputs into a MultiIndex DataFrame:
          index = (reinvest_rate, date)
          columns include draw-level series + repeated summary columns for convenience
        """
        frames = []
        for rate, out in outputs.items():
            dd = out["draw_detail"].copy()
            dd["reinvest_rate"] = float(rate)

            # Attach summary fields as constant columns (handy for groupby/selection)
            dd["roi"] = float(out.get("roi", np.nan))
            dd["net_profit_final"] = float(out.get("net_profit", np.nan))
            dd["final_equity"] = float(out.get("final_equity", np.nan))
            dd["total_contributed_final"] = float(out.get("total_contributed", np.nan))

            frames.append(dd)

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["reinvest_rate", "date"]).sort_index()
        return df

    def _compare_fixed_exposure(
        self,
        reinvest_rates: Tuple[float, ...],
        *,
        rng: Optional[np.random.Generator],
    ) -> Dict[float, Dict[str, Any]]:
        """
        Counterfactual comparison: identical ticket purchases per draw across rates.

        Implementation:
          - For each draw, generate tickets using ONLY ticket_budget (fixed exposure).
          - Compute draw_payout series once.
          - For each rate, apply reinvest/withdraw accounting to that same payout series.

        Result:
          - Spikes align exactly across reinvest rates.
          - Differences reflect only the accounting choice (withdraw vs keep as bankroll cash).
          - Note: since reinvested cash does not buy more tickets in this mode, higher reinvest
            will generally *not* improve outcomes (by design).
        """
        r_draw = (1.0 + self.withdrawal_apy) ** (1.0 / self.draws_per_year)

        # Generate canonical payouts once
        canonical = []
        for i in range(len(self.draws)):
            draw_pp = int(self._power_play_mult[i])
            batch = self._generate_tickets_for_budget(
                int(self.ticket_budget),
                draw_power_play=draw_pp,
                rng=rng,  # recommended for determinism in this mode
            )
            actual_spend = float(batch.costs.sum()) if batch.costs.size > 0 else 0.0

            payouts = self._score_tickets(
                whites=batch.whites,
                reds=batch.reds,
                mults=batch.multipliers,
                win_whites=self._win_whites[i],
                win_red=int(self._win_reds[i]),
                jackpot_value=float(self._jackpot_values[i]),
            )
            canonical.append({
                "date": self._draw_dates[i],
                "spend": actual_spend,
                "draw_payout": float(payouts.sum()),
            })

        canonical_df = pd.DataFrame(canonical)
        canonical_df["draw_net"] = canonical_df["draw_payout"] - canonical_df["spend"]
        canonical_df["contributed"] = self.ticket_budget * (np.arange(len(canonical_df)) + 1)

        outputs: Dict[float, Dict[str, Any]] = {}
        growth = self._compute_withdrawal_growth(canonical_df["date"])

        for rate in reinvest_rates:
            withdrawn_balance = 0.0
            bankroll_cash = 0.0

            rows = []
            for j in range(len(canonical_df)):
                draw_payout = float(canonical_df.loc[j, "draw_payout"])
                spend = float(canonical_df.loc[j, "spend"])
                contributed = float(canonical_df.loc[j, "contributed"])

                reinvested = float(draw_payout * float(rate))
                withdrawn = float(draw_payout - reinvested)

                withdrawn_balance = withdrawn_balance * float(growth[j]) + withdrawn
                bankroll_cash = bankroll_cash + reinvested  # not spent in fixed-exposure mode

                equity = withdrawn_balance + bankroll_cash
                net_profit = equity - contributed

                rows.append({
                    "date": canonical_df.loc[j, "date"],
                    "available": float(self.ticket_budget),  # fixed-exposure notion
                    "spend": spend,
                    "draw_payout": draw_payout,
                    "draw_net": draw_payout - spend,
                    "reinvested": reinvested,
                    "withdrawn": withdrawn,
                    "withdrawn_balance": withdrawn_balance,
                    "bankroll_cash": bankroll_cash,
                    "equity": equity,
                    "contributed": contributed,
                    "net_profit": net_profit,
                })

            draw_df = pd.DataFrame(rows)
            draw_df = self._add_rolling_metrics(draw_df)

            final_equity = float(draw_df["equity"].iloc[-1]) if not draw_df.empty else 0.0
            total_contributed = float(draw_df["contributed"].iloc[-1]) if not draw_df.empty else 0.0
            roi = final_equity / total_contributed if total_contributed > 0 else np.nan

            outputs[float(rate)] = {
                "draw_detail": draw_df,
                "net_profit": float(draw_df["net_profit"].iloc[-1]) if not draw_df.empty else 0.0,
                "roi": float(roi),
                "final_equity": float(final_equity),
                "total_contributed": float(total_contributed),
            }

        return outputs

    def _compare_nested_compounding(
        self,
        reinvest_rates: Tuple[float, ...],
        *,
        rng: Optional[np.random.Generator],
    ) -> Dict[float, Dict[str, Any]]:
        """
        Compounding exposure comparison with *shared, nested ticket pools* per draw.

        Key property:
          - For each draw, we generate enough multiplier tickets and non-mult tickets to cover
            the maximum demand across strategies that draw.
          - Each strategy takes the first n_mult and first n_no tickets from the respective pools.
          - Therefore, higher-spend strategies strictly include lower-spend tickets, draw-by-draw,
            eliminating the “50% spikes but 100% doesn’t” anomaly caused by different random tickets.

        This is the mode to use if you want “reinvest => buy more tickets” and you still want
        comparisons that respect shared randomness.
        """
        growth = self._compute_withdrawal_growth(self._draw_dates)

        # Per-rate state
        state = {
            float(rate): {
                "withdrawn_balance": 0.0,
                "bankroll_cash": 0.0,
                "contributed": 0.0,
                "rows": [],
            }
            for rate in reinvest_rates
        }

        for i in range(len(self.draws)):
            # Each strategy gets baseline contribution each draw
            for rate in reinvest_rates:
                state[float(rate)]["contributed"] += self.ticket_budget

            # Determine each strategy's available dollars and desired ticket counts
            needs = {}
            for rate in reinvest_rates:
                r = float(rate)
                available = float(self.ticket_budget) + float(state[r]["bankroll_cash"])
                budget_int = int(available)
                n_mult, n_no = self._allocate_ticket_counts(budget_int)
                needs[r] = {
                    "available": available,
                    "budget_int": budget_int,
                    "n_mult": n_mult,
                    "n_no": n_no,
                }

            max_n_mult = max(needs[r]["n_mult"] for r in needs)
            max_n_no = max(needs[r]["n_no"] for r in needs)

            # Generate shared pools once per draw (fixed order for determinism)
            # Multiplier pool
            if max_n_mult > 0:
                mult_batch = self._generate_batch(
                    max_n_mult,
                    include_metadata=True,
                    rng=rng,
                )
                mult_whites = np.empty((max_n_mult, 5), dtype=np.int64)
                mult_reds = np.empty((max_n_mult,), dtype=np.int64)
                for j, t in enumerate(mult_batch):
                    w = t["white_balls"]
                    mult_whites[j, :] = (int(w[0]), int(w[1]), int(w[2]), int(w[3]), int(w[4]))
                    mult_reds[j] = int(t["red_ball"])
                draw_pp = int(self._power_play_mult[i])
                mult_mults = np.full((max_n_mult,), draw_pp, dtype=np.int16)
                mult_payouts = self._score_tickets(
                    whites=mult_whites,
                    reds=mult_reds,
                    mults=mult_mults,
                    win_whites=self._win_whites[i],
                    win_red=int(self._win_reds[i]),
                    jackpot_value=float(self._jackpot_values[i]),
                )
            else:
                mult_payouts = np.empty((0,), dtype=np.float64)

            # Non-mult pool
            if max_n_no > 0:
                no_batch = self._generate_batch(
                    max_n_no,
                    include_metadata=True,
                    rng=rng,
                    existing_tickets=mult_batch,
                )
                no_whites = np.empty((max_n_no, 5), dtype=np.int64)
                no_reds = np.empty((max_n_no,), dtype=np.int64)
                for j, t in enumerate(no_batch):
                    w = t["white_balls"]
                    no_whites[j, :] = (int(w[0]), int(w[1]), int(w[2]), int(w[3]), int(w[4]))
                    no_reds[j] = int(t["red_ball"])
                no_mults = np.ones((max_n_no,), dtype=np.int16)  
                no_payouts = self._score_tickets(
                    whites=no_whites,
                    reds=no_reds,
                    mults=no_mults,
                    win_whites=self._win_whites[i],
                    win_red=int(self._win_reds[i]),
                    jackpot_value=float(self._jackpot_values[i]),
                )
            else:
                no_payouts = np.empty((0,), dtype=np.float64)

            # Apply each strategy's selection + update wealth state
            for rate in reinvest_rates:
                r = float(rate)

                n_mult = int(needs[r]["n_mult"])
                n_no = int(needs[r]["n_no"])
                available = float(needs[r]["available"])

                draw_payout = float(mult_payouts[:n_mult].sum() + no_payouts[:n_no].sum())
                actual_spend = float(3 * n_mult + 2 * n_no)

                reinvested = float(draw_payout * r)
                withdrawn = float(draw_payout - reinvested)

                withdrawn_balance = float(state[r]["withdrawn_balance"]) * float(growth[i]) + withdrawn
                bankroll_cash = (available - actual_spend) + reinvested

                equity = withdrawn_balance + bankroll_cash
                contributed = float(state[r]["contributed"])
                net_profit = equity - contributed
                draw_net = draw_payout - actual_spend

                state[r]["withdrawn_balance"] = withdrawn_balance
                state[r]["bankroll_cash"] = bankroll_cash

                state[r]["rows"].append({
                    "date": self._draw_dates[i],
                    "available": available,
                    "spend": actual_spend,
                    "draw_payout": draw_payout,
                    "draw_net": draw_net,
                    "reinvested": reinvested,
                    "withdrawn": withdrawn,
                    "withdrawn_balance": withdrawn_balance,
                    "bankroll_cash": bankroll_cash,
                    "equity": equity,
                    "contributed": contributed,
                    "net_profit": net_profit,
                })

        outputs: Dict[float, Dict[str, Any]] = {}
        for rate in reinvest_rates:
            r = float(rate)
            draw_df = pd.DataFrame(state[r]["rows"])
            draw_df = self._add_rolling_metrics(draw_df)

            final_equity = float(draw_df["equity"].iloc[-1]) if not draw_df.empty else 0.0
            total_contributed = float(draw_df["contributed"].iloc[-1]) if not draw_df.empty else 0.0
            roi = final_equity / total_contributed if total_contributed > 0 else np.nan

            outputs[r] = {
                "draw_detail": draw_df,
                "net_profit": float(draw_df["net_profit"].iloc[-1]) if not draw_df.empty else 0.0,
                "roi": float(roi),
                "final_equity": float(final_equity),
                "total_contributed": float(total_contributed),
            }

        return outputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, seed: Optional[int] = None) -> pd.DataFrame:
        """Run a single strategy path using the compounding exposure model.

        If seed is None, uses self.seed. Otherwise uses provided seed for this run only.

        Determinism / RNG isolation
        ---------------------------
        The ticket generator currently relies on NumPy's global RNG (`np.random.*`).
        This method temporarily sets the global RNG seed and restores the prior state on exit.

        Returns: per-draw pd.DataFrame (equity, spend, draw_payout, net_profit, etc.)
        Side effects: sets self.pnl_table, self.last_ticket_detail, self.last_summary, self.last_seed_used
        """
        seed_to_use = self.seed if seed is None else int(seed)

        rng = np.random.default_rng(seed_to_use)

        with _temporary_numpy_seed(seed_to_use):
            draw_df, ticket_df, summary = self._replay_single(rng=rng)

        self.pnl_table = draw_df[
            [
                "date",
                "equity",
                "net_profit",
                "contributed",
                "available",
                "spend",
                "draw_payout",
                "withdrawn",
                "withdrawn_balance",
                "bankroll_cash",
            ]
        ].copy()

        self.last_ticket_detail = ticket_df
        self.last_summary = summary
        self.last_seed_used = int(seed_to_use)
    
        return draw_df


    def compare_reinvest_rates(
        self,
        reinvest_rates: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
        seed: Optional[int] = None,
        plot: bool = True,
        withdrawal_apy: Optional[float] = None,
        draws_per_year: Optional[int] = None,
        mode: str = "nested_compounding",
    ) -> pd.DataFrame:
        """Compare multiple reinvest rates and return a MultiIndex DataFrame.

        Modes
        -----
        fixed_exposure:
          - Same tickets purchased per draw for all rates (tickets based on ticket_budget only).
          - Spikes align; this isolates accounting effects but does NOT model “reinvest buys more tickets”.

        nested_compounding:
          - Models “reinvest buys more tickets”, but ensures fairness by generating shared nested
            ticket pools per draw (higher spend includes the lower spend ticket set).

        Args:
            reinvest_rates: tuple of reinvest fractions in [0, 1].
            seed: optional seed. If None, uses self.seed.
            plot: if True, render a comparison plot.
            withdrawal_apy: optional override for this comparison only.
            draws_per_year: optional override for this comparison only.
            mode: 'fixed_exposure' or 'nested_compounding'.

        Returns:
            MultiIndex DataFrame indexed by (reinvest_rate, date).
        """
        seed_to_use = self.seed if seed is None else int(seed)

        original_apy = float(self.withdrawal_apy)
        original_dpy = int(self.draws_per_year)

        try:
            if withdrawal_apy is not None:
                self.withdrawal_apy = float(withdrawal_apy)
            if draws_per_year is not None:
                self.draws_per_year = int(draws_per_year)

            rng = np.random.default_rng(seed_to_use)

            with _temporary_numpy_seed(seed_to_use):
                if mode == "fixed_exposure":
                    outputs = self._compare_fixed_exposure(reinvest_rates=reinvest_rates, rng=rng)
                elif mode == "nested_compounding":
                    outputs = self._compare_nested_compounding(reinvest_rates=reinvest_rates, rng=rng)
                else:
                    raise ValueError("mode must be 'fixed_exposure' or 'nested_compounding'")

        finally:
            self.withdrawal_apy = original_apy
            self.draws_per_year = original_dpy

        if plot:
            self._plot_compare_reinvest(outputs)

        return self._comparison_outputs_to_df(outputs)

    @staticmethod
    def _format_halfyear_ticks(ax) -> None:
        """
        Configure x-axis ticks to show half-year starts only:
          - Jan 1 => 'YYYY-1H'
          - Jul 1 => 'YYYY-2H'
        Assumes x-data are datetime-like.
        """
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7], bymonthday=1))

        def _fmt(x, pos=None):
            dt = mdates.num2date(x)
            if dt.month == 1:
                return f"{dt.year}-1H"
            if dt.month == 7:
                return f"{dt.year}-2H"
            return ""

        ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    @staticmethod
    def _marker_color_for_payout(total_payout: float) -> Optional[str]:
        """
        Only show significant wins:
          - Yellow: 1,000 to 20,000
          - Red: > 20,000
        Suppress small wins (< 1,000) to reduce clutter.
        """
        if total_payout > 20_000:
            return "red"
        if total_payout >= 1_000:
            return "gold"  # more legible than pure yellow in many themes
        return None

    def plot_winnings(self, draw_detail: pd.DataFrame) -> None:
        """
        More intuitive visualization suite.

        Panel 1: Equity vs Contributed (with shaded Net Profit region)
        Panel 2: Net Profit + Rolling Net Profit; secondary axis for rolling $/draw
        Panel 3: Per-draw cashflows (Spend vs Payout)

        Annotations:
          - Only significant win draws (>= $1,000) get markers (gold/red).
        """
        import matplotlib.pyplot as plt

        # Backward compatible: accept dict returned by older run()
        if isinstance(draw_detail, dict):
            draw_detail = draw_detail.get("draw_detail") or draw_detail.get("pnl_table")
        
        if draw_detail is None or draw_detail.empty:
            raise ValueError("draw_detail is empty; run backtest first and pass the DataFrame returned by run().")

        dd = draw_detail.copy()
        x = pd.to_datetime(dd["date"])

        equity = dd["equity"].astype(float).to_numpy()
        contributed = dd["contributed"].astype(float).to_numpy()
        net_profit = dd["net_profit"].astype(float).to_numpy()
        spend = dd["spend"].astype(float).to_numpy()
        payout = dd["draw_payout"].astype(float).to_numpy()

        # Rolling trend: use net_profit delta (per draw P&L) and rolling mean of net_profit
        net_profit_delta = np.diff(net_profit, prepend=net_profit[0])
        roll_w = int(self.rolling_window)
        roll_minp = max(5, roll_w // 3)

        roll_np_delta = (
            pd.Series(net_profit_delta)
            .rolling(window=roll_w, min_periods=roll_minp)
            .mean()
            .to_numpy()
        )

        roll_net_profit = (
            pd.Series(net_profit)
            .rolling(window=roll_w, min_periods=roll_minp)
            .mean()
            .to_numpy()
        )

        # --- Size tweaks requested ---
        # Double the height of each subplot: previously (13,10) total height; now ~2x.
        # Increase width by 25%: 13 -> 16.25.
        fig, axes = plt.subplots(
            3, 1,
            figsize=(16.25, 20.0),  # wider + taller
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.6, 1.4]},
        )

        # -------------------------
        # Panel 1: Equity vs Contributed + shaded net profit
        # -------------------------
        ax0 = axes[0]
        ax0.plot(x, contributed, label="Contributed (your deposits)", linewidth=1.2)
        ax0.plot(x, equity, label="Equity (withdrawn acct + bankroll cash)", linewidth=1.6)

        above = equity >= contributed
        ax0.fill_between(x, contributed, equity, where=above, alpha=0.18, interpolate=True, label="Net Profit (positive)")
        ax0.fill_between(x, contributed, equity, where=~above, alpha=0.18, interpolate=True, label="Net Profit (negative)")

        ax0.set_title("Equity vs Contributed (Shaded Region = Net Profit)")
        ax0.set_ylabel("Dollars")
        ax0.axhline(0, linestyle="--", linewidth=1)

        # Significant win markers on equity curve
        for xi, yi, pi in zip(x, equity, payout):
            c = self._marker_color_for_payout(float(pi))
            if c is None:
                continue
            ax0.scatter([xi], [yi], s=22, marker="v", c=c, edgecolors="none", zorder=5)

        ax0.legend(loc="upper left", frameon=False, ncol=1)

        # -------------------------
        # Panel 2: Net Profit + rolling net profit
        # -------------------------
        ax1 = axes[1]
        ax1.plot(
            x, net_profit,
            label="Net Profit (equity - contributed) [left axis: $]",
            linewidth=1.3,
        )
        ax1.plot(
            x, roll_net_profit,
            label=f"Rolling Net Profit (mean, window={roll_w}) [left axis: $]",
            linewidth=1.3,
        )

        # 0 reference for LEFT axis ($)
        ax1.axhline(0, linestyle="--", linewidth=1)
        ax1.set_title("Net Profit Level (Left) vs Average Change per Draw (Right)")
        ax1.set_ylabel("Dollars ($)")

        ax1b = ax1.twinx()
        ax1b.plot(
            x, roll_np_delta,
            label=f"Rolling Profit/Draw (mean Δ, window={roll_w}) [right axis: $/draw]",
            linewidth=1.0,
        )

        # 0 reference for RIGHT axis ($/draw)
        ax1b.axhline(0, linestyle=":", linewidth=1)
        ax1b.set_ylabel("$/draw")

        # Combined legend (fixed location, no loc='best')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax1b.get_legend_handles_labels()
        ax1b.legend(h1 + h2, l1 + l2, loc="lower left", frameon=False)

        # -------------------------
        # Panel 3: Spend vs Payout cashflows
        # -------------------------
        ax2 = axes[2]
        ax2.bar(x, -spend, width=2.0, align="center", label="Spend (tickets)", alpha=0.45)
        ax2.bar(x, payout, width=2.0, align="center", label="Payout (winnings)", alpha=0.45)

        ax2.axhline(0, linestyle="--", linewidth=1)
        ax2.set_title("Per-Draw Cashflows (Spend vs Payout)")
        ax2.set_ylabel("Dollars")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left", frameon=False)

        # Half-year ticks: 1H / 2H
        self._format_halfyear_ticks(ax2)

        # X-label rotation: 45 degrees (prevents overlap while remaining readable)
        fig.autofmt_xdate(rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    def _plot_compare_reinvest(self, outputs: Dict[float, Dict[str, Any]]) -> None:
        """
        Cleaner comparison plot: net profit lines, fixed legend loc, 1H/2H ticks.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(16.25, 6.25))  # width +25%, height bumped for readability

        for rate, out in outputs.items():
            df = out["draw_detail"]
            x = pd.to_datetime(df["date"])
            ax.plot(x, df["net_profit"], label=f"Reinvest {int(rate * 100)}%")

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title("Net Profit Over Time by Reinvestment Rate")
        ax.set_xlabel("Date")
        ax.set_ylabel("Dollars")
        ax.legend(loc="upper left", frameon=False, ncol=2)

        self._format_halfyear_ticks(ax)
        fig.autofmt_xdate(rotation=45, ha="right")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Stratified analysis helper
    # ------------------------------------------------------------------
    def summarize_by_white_temperature_deciles(self, ticket_detail: pd.DataFrame, q: int = 10) -> pd.DataFrame:
        """
        Stratify ticket performance by white temperature deciles.
        Avoids pandas FutureWarning by using observed=True.
        """
        if "white_temperature" not in ticket_detail.columns:
            raise ValueError("ticket_detail does not include white_temperature. Ensure store_temperatures=True.")

        df = ticket_detail.copy()
        df["temp_bin"] = pd.qcut(df["white_temperature"], q)

        summary = (
            df.groupby("temp_bin", observed=True)
            .agg(
                mean_payout=("payout", "mean"),
                hit_rate=("payout", lambda x: (x > 0).mean()),
                n=("payout", "size"),
            )
        )
        return summary

# ---------------------------------------------------------------------
# Minimal unit tests (optional)
# ---------------------------------------------------------------------
def run_unit_tests() -> None:
    """Run a minimal self-test suite.

    These tests are intentionally lightweight and require no external test runner.
    They can be executed via:

        python powerball_backtester.py

    Notes:
      - Some integration tests (run determinism) are only executed if the default
        CSV files are discoverable relative to the current working directory.
    """

    def _assert(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # --- Allocation edge cases (remainder handling) ---
    bt = PowerballBacktester.__new__(PowerballBacktester)  # avoid __init__
    bt.use_multiplier = True

    n_mult, n_no = PowerballBacktester._allocate_ticket_counts(bt, 10)  # 10 = 3+3+2+2
    _assert((n_mult, n_no) == (2, 2), f"allocate(10) expected (2,2) got {(n_mult,n_no)}")

    n_mult, n_no = PowerballBacktester._allocate_ticket_counts(bt, 4)  # avoid remainder=1 => 2+2
    _assert((n_mult, n_no) == (0, 2), f"allocate(4) expected (0,2) got {(n_mult,n_no)}")

    n_mult, n_no = PowerballBacktester._allocate_ticket_counts(bt, 7)  # 3+2+2
    _assert((n_mult, n_no) == (1, 2), f"allocate(7) expected (1,2) got {(n_mult,n_no)}")

    # --- Scoring correctness + kernel parity ---
    win_whites = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    win_red = 10
    jackpot_value = 100_000_000.0

    ticket_whites = np.array(
        [
            [1, 2, 3, 4, 5],   # jackpot
            [1, 2, 3, 4, 5],   # 5 whites (no red)
            [1, 2, 3, 4, 99],  # 4+red
            [1, 2, 3, 99, 98], # 3+red
            [1, 2, 99, 98, 97],# 2+red
            [1, 99, 98, 97, 96],# 1+red
            [99, 98, 97, 96, 95],# red only
            [99, 98, 97, 96, 95],# nothing
        ],
        dtype=np.int64,
    )
    ticket_reds = np.array([10, 11, 10, 10, 10, 10, 10, 11], dtype=np.int64)
    # Multipliers: 1 means no Power Play; otherwise the draw's Power Play multiplier (2/3/4/5/10).
    # Jackpot is never multiplied.
    multipliers = np.array([1, 10, 2, 1, 2, 1, 2, 1], dtype=np.int16)

    p_numpy = _score_kernel_numpy(ticket_whites, ticket_reds, multipliers, win_whites, win_red, jackpot_value)
    p_numba = _score_kernel_numba(ticket_whites, ticket_reds, multipliers, win_whites, win_red, jackpot_value)

    _assert(np.allclose(p_numpy, p_numba), "NumPy and Numba kernels disagree")

    expected = np.array(
        [
            jackpot_value,                 # jackpot (no multiplier)
            PRIZE_5_WHITE * 10.0,         # 5 whites with multiplier (tests 10x)
            PRIZE_4_WHITE_RED * 2.0,      # 4+red with multiplier
            PRIZE_3_WHITE_RED * 1.0,      # 3+red no multiplier
            PRIZE_2_WHITE_RED * 2.0,      # 2+red with multiplier
            PRIZE_1_WHITE_RED * 1.0,      # 1+red no multiplier
            PRIZE_RED_ONLY * 2.0,         # red only with multiplier
            0.0,
        ],
        dtype=np.float64,
    )
    _assert(np.allclose(p_numpy, expected), "Scoring does not match expected payouts")

    # --- RNG isolation smoke test (global RNG state restored) ---
    np.random.seed(123)
    a = np.random.randint(0, 1_000_000)
    with _temporary_numpy_seed(999):
        _ = np.random.randint(0, 1_000_000)
    b = np.random.randint(0, 1_000_000)
    np.random.seed(123)
    a2 = np.random.randint(0, 1_000_000)
    b2 = np.random.randint(0, 1_000_000)
    _assert((a, b) == (a2, b2), "_temporary_numpy_seed did not restore global RNG state")

    print("All unit tests passed.")
