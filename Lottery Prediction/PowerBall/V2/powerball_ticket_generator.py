import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union, Literal

import numpy as np
import pandas as pd


TicketLike = Union[
    Tuple[int, int, int, int, int, int],  # (w1,w2,w3,w4,w5,red) with whites sorted
    Dict[str, Any],                        # either {"white_balls":[...],"red_ball":...} or flat keys
]


@dataclass(frozen=True)
class Ticket:
    """
    Canonical ticket representation used for uniqueness checks.

    Whites are stored sorted (w1<w2<w3<w4<w5).
    """
    w1: int
    w2: int
    w3: int
    w4: int
    w5: int
    red: int

    @property
    def as_tuple(self) -> Tuple[int, int, int, int, int, int]:
        return (self.w1, self.w2, self.w3, self.w4, self.w5, self.red)



    @staticmethod
    def _from_parts(whites: Sequence[int], red: int) -> "Ticket":
        """
        Construct a Ticket from raw parts with validation and canonicalization.

        - Whites are sorted into ascending order.
        - Validation enforces exactly 5 unique whites in [1, 69] and red in [1, 26].
        """
        if len(whites) != 5:
            raise ValueError(f"Ticket must have exactly 5 white balls; got {len(whites)}")

        try:
            whites_i = [int(w) for w in whites]
            red_i = int(red)
        except Exception as e:
            raise ValueError(f"Ticket parts must be integers: whites={whites!r}, red={red!r}") from e

        if any(w < 1 or w > 69 for w in whites_i):
            raise ValueError(f"White balls must be in [1, 69]; got {whites_i!r}")
        if len(set(whites_i)) != 5:
            raise ValueError(f"White balls must be unique; got {whites_i!r}")
        if red_i < 1 or red_i > 26:
            raise ValueError(f"Red ball must be in [1, 26]; got {red_i!r}")

        whites_sorted = sorted(whites_i)
        return Ticket(whites_sorted[0], whites_sorted[1], whites_sorted[2], whites_sorted[3], whites_sorted[4], red_i)
    @staticmethod
    def from_any(x: TicketLike) -> "Ticket":
        """
        Normalize a ticket into the canonical internal representation used for uniqueness checks.

        Accepted inputs
        ---------------
        - Tuple[int,int,int,int,int,int]: (w1,w2,w3,w4,w5,red) (whites may be unsorted)
        - Dict with nested schema: {"white_balls": [...5...], "red_ball": r}
        - Dict with flat schema:   {"white_1":..,"white_5":..,"red_ball": r}

        Validation
        ----------
        - Whites: exactly 5 integers, unique, each in [1, 69]
        - Red: integer in [1, 26]

        Raises
        ------
        ValueError
            If the input cannot be parsed or violates invariants.
        TypeError
            If the input type is unsupported.
        """
        if isinstance(x, Ticket):
            return x

        whites: List[int]
        red: int

        if isinstance(x, tuple) and len(x) == 6:
            w1, w2, w3, w4, w5, red = map(int, x)
            whites = [w1, w2, w3, w4, w5]
            return Ticket._from_parts(whites, red)

        if isinstance(x, dict):
            # nested schema: {"white_balls":[...], "red_ball":...}
            if "white_balls" in x and "red_ball" in x:
                whites = list(map(int, x["white_balls"]))
                red = int(x["red_ball"])
                return Ticket._from_parts(whites, red)

            # flat schema: white_1..white_5, red_ball
            flat_keys = ("white_1", "white_2", "white_3", "white_4", "white_5", "red_ball")
            if all(k in x for k in flat_keys):
                whites = [int(x["white_1"]), int(x["white_2"]), int(x["white_3"]), int(x["white_4"]), int(x["white_5"])]
                red = int(x["red_ball"])
                return Ticket._from_parts(whites, red)

            raise ValueError(
                "Unsupported ticket dict schema. Expected either "
                "{'white_balls': [...], 'red_ball': ...} or flat keys "
                "white_1..white_5 plus 'red_ball'."
            )

        raise TypeError("Unsupported ticket representation for uniqueness checks.")


class TemperatureLotteryGenerator:
    """
    Temperature-controlled generator for Powerball white and red balls.

    Legacy behavior (default):
      alpha = clip(T / max_T, 0, 1)
      T sampled uniformly from [T_min, max_T] when T_white/T_red are None

    New opt-in behavior (recommended for your intent):
      alpha = clip(T / temperature_scale, 0, 1)   # temperature_scale is fixed (e.g., 200.0)
      T sampled from a configurable distribution over [T_min, max_T]
        - "rev_log1p" heavily favors high T (mostly random) with a small tail near 0 (frequency-based)

    Key features added:
      - supports max_T == 0 (forces T=0, i.e., empirical distribution)
      - temperature_scale (fixed mapping from T->alpha so max_T acts like a cap, not a rescaling)
      - temperature_sampling mode: "uniform" (legacy), "log1p" (skew low), "rev_log1p" (skew high)
    """

    TemperatureSampling = Literal["uniform", "log1p", "rev_log1p"]

    def __init__(
        self,
        csv_path: str = "powerball.csv",
        *,
        df: Optional[pd.DataFrame] = None,
        T_white_min: float = 35.0,
        T_red_min: float = 20.0,
        smoothing: float = 1.0,
        temperature_scale: Optional[float] = None,
        temperature_sampling: TemperatureSampling = "uniform",
    ) -> None:
        if df is None:
            df = pd.read_csv(csv_path)
        else:
            df = df.copy()

        if "white_balls" not in df.columns or "red_ball" not in df.columns:
            raise ValueError("csv_path must contain columns: 'white_balls' and 'red_ball'")

        # ----- White balls (1..69) -----
        white = df["white_balls"].astype(str).str.split("|", expand=True).astype(int)
        white_vals = np.arange(1, 70, dtype=np.int64)
        w_counts = np.zeros_like(white_vals, dtype=np.float64)

        flat_white = white.values.reshape(-1)
        for v in flat_white:
            iv = int(v)
            if 1 <= iv <= 69:
                w_counts[iv - 1] += 1.0

        w_counts += float(smoothing)
        w_probs = w_counts / w_counts.sum()

        # ----- Red balls (1..26) -----
        red_vals = np.arange(1, 27, dtype=np.int64)
        r_counts = np.zeros_like(red_vals, dtype=np.float64)

        for v in df["red_ball"].astype(int).values:
            iv = int(v)
            if 1 <= iv <= 26:
                r_counts[iv - 1] += 1.0

        r_counts += float(smoothing)
        r_probs = r_counts / r_counts.sum()

        self.white_vals = white_vals
        self.red_vals = red_vals
        self._white_empirical = w_probs
        self._red_empirical = r_probs

        self.T_white_min = float(T_white_min)
        self.T_red_min = float(T_red_min)

        # If set, alpha = T / temperature_scale (clipped). If None, legacy alpha = T / max_T.
        self.temperature_scale = None if temperature_scale is None else float(temperature_scale)
        self.temperature_sampling: TemperatureLotteryGenerator.TemperatureSampling = temperature_sampling

    @staticmethod
    def _rng_from(rng: Optional[np.random.Generator], seed: Optional[int]) -> np.random.Generator:
        if rng is not None:
            return rng
        return np.random.default_rng(seed)

    @staticmethod
    def _alpha(*, T: float, max_T: float, temperature_scale: Optional[float]) -> float:
        """
        Map temperature to mixture coefficient alpha in [0,1].
        - If temperature_scale is None => legacy alpha = T/max_T
        - Else alpha = T/temperature_scale
        For scale<=0 we define alpha=0 (pure empirical).
        """
        scale = float(max_T) if temperature_scale is None else float(temperature_scale)
        if scale <= 0.0:
            return 0.0
        a = float(T) / scale
        if a < 0.0:
            return 0.0
        if a > 1.0:
            return 1.0
        return a

    @staticmethod
    def _mix_probs(
        empirical: np.ndarray,
        *,
        T: float,
        max_T: float,
        temperature_scale: Optional[float],
    ) -> np.ndarray:
        # For max_T==0 (or scale==0), alpha=0 => empirical.
        alpha = TemperatureLotteryGenerator._alpha(T=T, max_T=max_T, temperature_scale=temperature_scale)
        u = np.full_like(empirical, 1.0 / empirical.size, dtype=np.float64)
        p = (1.0 - alpha) * empirical + alpha * u

        # numerical guard
        p = np.maximum(p, 0.0)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            return u
        return p / s

    @staticmethod
    def _sample_T(
        rng: np.random.Generator,
        *,
        low: float,
        high: float,
        mode: "TemperatureLotteryGenerator.TemperatureSampling",
    ) -> float:
        """
        Sample T in [low, high].
          - uniform:   Uniform(low, high)
          - log1p:     skew toward low (defined at 0)
          - rev_log1p: skew toward high (mostly high T, a few low T)

        Note: "log1p"/"rev_log1p" are stable at 0 and do not require log(0).
        """
        low = float(low)
        high = float(high)

        if high <= 0.0:
            return 0.0
        if low > high:
            low = high
        if low == high:
            return low

        span = high - low
        u = float(rng.uniform(0.0, 1.0))

        if mode == "uniform":
            return float(low + u * span)

        z = float(np.expm1(u * np.log1p(span)))  # z in [0, span], skewed to 0
        if mode == "log1p":
            return float(low + z)
        if mode == "rev_log1p":
            return float(high - z)

        raise ValueError(f"Unknown temperature_sampling mode: {mode!r}")

    def generate_ticket_batch(
        self,
        n: int,
        *,
        max_T: float = 100.0,
        include_metadata: bool = True,
        # Optional overrides (if None, sample from [T_min, max_T] according to temperature_sampling)
        T_white: Optional[float] = None,
        T_red: Optional[float] = None,
        # Uniqueness controls
        ensure_unique: bool = True,
        existing_tickets: Optional[Iterable[TicketLike]] = None,
        max_rounds: int = 50,
        # RNG controls
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate `n` tickets.

        Returns
        -------
        List[dict]
            If include_metadata=True:
              {"white_balls":[w1..w5], "red_ball": r, "T_white":..., "T_red":..., "max_T":...}
            If include_metadata=False:
              {"white_1":..., ..., "white_5":..., "red_ball":...}

        Notes on uniqueness
        -------------------
        If ensure_unique=True, returned tickets are unique with respect to (sorted whites, red).
        If `existing_tickets` is provided, uniqueness is enforced against that set as well.

        If uniqueness cannot be satisfied within `max_rounds`, this method raises RuntimeError.
        (It does not silently return fewer than `n` tickets.)

        RNG precedence
        --------------
        If `rng` is provided, it is used directly and `seed` is ignored.
        Otherwise, a new numpy Generator is created from `seed` (or from the instance seed).
        """
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []

        max_T = float(max_T)
        if max_T < 0:
            raise ValueError("max_T must be >= 0")

        rng_ = self._rng_from(rng, seed)

        # Seed uniqueness set
        seen: Set[Tuple[int, int, int, int, int, int]] = set()
        if existing_tickets is not None:
            for t in existing_tickets:
                seen.add(Ticket.from_any(t).as_tuple)

        out: List[Dict[str, Any]] = []
        rounds = 0

        def _sample_one() -> Tuple[Ticket, Dict[str, Any]]:
            # Temperatures: support max_T == 0 => force T=0 (pure empirical)
            if max_T == 0.0:
                Tw = 0.0
                Tr = 0.0
            else:
                Tw = (
                    float(T_white)
                    if T_white is not None
                    else self._sample_T(
                        rng_,
                        low=min(self.T_white_min, max_T),
                        high=max_T,
                        mode=self.temperature_sampling,
                    )
                )
                Tr = (
                    float(T_red)
                    if T_red is not None
                    else self._sample_T(
                        rng_,
                        low=min(self.T_red_min, max_T),
                        high=max_T,
                        mode=self.temperature_sampling,
                    )
                )

            # Clip to [0, max_T]
            Tw = max(0.0, min(float(Tw), max_T))
            Tr = max(0.0, min(float(Tr), max_T))

            p_w = self._mix_probs(
                self._white_empirical,
                T=Tw,
                max_T=max_T,
                temperature_scale=self.temperature_scale,
            )
            p_r = self._mix_probs(
                self._red_empirical,
                T=Tr,
                max_T=max_T,
                temperature_scale=self.temperature_scale,
            )

            whites = rng_.choice(self.white_vals, size=5, replace=False, p=p_w).astype(int)
            whites.sort()
            red = int(rng_.choice(self.red_vals, size=1, replace=True, p=p_r)[0])

            ticket = Ticket(int(whites[0]), int(whites[1]), int(whites[2]), int(whites[3]), int(whites[4]), red)

            if include_metadata:
                payload: Dict[str, Any] = {
                    "white_balls": [ticket.w1, ticket.w2, ticket.w3, ticket.w4, ticket.w5],
                    "red_ball": ticket.red,
                    "T_white": Tw,
                    "T_red": Tr,
                    "max_T": max_T,
                }
            else:
                payload = {
                    "white_1": ticket.w1,
                    "white_2": ticket.w2,
                    "white_3": ticket.w3,
                    "white_4": ticket.w4,
                    "white_5": ticket.w5,
                    "red_ball": ticket.red,
                }
            return ticket, payload

        if not ensure_unique:
            for _ in range(n):
                _, payload = _sample_one()
                out.append(payload)
            return out

        # ensure_unique=True: iterative fill, resampling only missing count
        while len(out) < n:
            rounds += 1
            if rounds > int(max_rounds):
                raise RuntimeError(
                    f"Unable to generate {n} unique tickets within max_rounds={max_rounds}. "
                    "Try increasing max_rounds or disabling ensure_unique."
                )

            remaining = n - len(out)

            # Oversample to reduce rounds; keep bounded.
            k = int(min(max(remaining, 1) * 2, max(remaining, 1) + 5000))

            for _ in range(k):
                if len(out) >= n:
                    break
                ticket, payload = _sample_one()
                key = ticket.as_tuple
                if key in seen:
                    continue
                seen.add(key)
                out.append(payload)

        return out

