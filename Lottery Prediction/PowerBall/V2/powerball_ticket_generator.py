import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

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
    def from_any(x: TicketLike) -> "Ticket":
        if isinstance(x, tuple) and len(x) == 6:
            w1, w2, w3, w4, w5, red = map(int, x)
            whites = sorted((w1, w2, w3, w4, w5))
            return Ticket(*whites, int(red))

        if isinstance(x, dict):
            if "white_balls" in x and "red_ball" in x:
                whites = list(map(int, x["white_balls"]))
                if len(whites) != 5:
                    raise ValueError("ticket dict white_balls must have length 5")
                whites = sorted(whites)
                return Ticket(*whites, int(x["red_ball"]))

            # flat schema: white_1..white_5, red_ball
            if all(k in x for k in ("white_1", "white_2", "white_3", "white_4", "white_5", "red_ball")):
                whites = sorted([int(x["white_1"]), int(x["white_2"]), int(x["white_3"]), int(x["white_4"]), int(x["white_5"])])
                return Ticket(*whites, int(x["red_ball"]))

        raise TypeError("Unsupported ticket representation for uniqueness checks.")


class TemperatureLotteryGenerator:
    """
    Temperature-controlled generator for Powerball white and red balls.

    Conceptual model
    ----------------
    The generator starts from empirical ball-frequency distributions derived from `csv_path`.
    It then uses "temperature" as a dial between:

      - Low temperature => distribution close to empirical frequencies (more mass on common numbers)
      - High temperature => flatter distribution (closer to uniform / "true random")

    In this implementation, we map temperature to a mixture coefficient:

        alpha = clip(T / max_T, 0, 1)
        p(T) = (1 - alpha) * p_empirical + alpha * p_uniform

    Therefore:
      - T = 0         => empirical
      - T = max_T     => uniform
      - Larger max_T  => more "resolution" in the dial (same T produces smaller alpha)

    Parameters
    ----------
    csv_path:
        Path to draw-history CSV used to estimate empirical frequencies.
        Expected columns:
          - 'white_balls' pipe-delimited 5 ints (e.g., "3|14|27|45|62")
          - 'red_ball' int
    T_white_min / T_red_min:
        Minimum temperatures used when sampling per-ticket temperatures.
        For each ticket, T is sampled uniformly from [T_min, max_T] (clipped to max_T).
        If you want fixed temperatures, pass T_white and/or T_red into generate_ticket_batch().
    smoothing:
        Laplace smoothing added to ball counts to avoid zero probability for unseen numbers.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        T_white_min: float = 35.0,
        T_red_min: float = 20.0,
        smoothing: float = 1.0,
    ) -> None:
        df = pd.read_csv(csv_path)

        if "white_balls" not in df.columns or "red_ball" not in df.columns:
            raise ValueError("csv_path must contain columns: 'white_balls' and 'red_ball'")

        # ----- White balls (1..69) -----
        white = df["white_balls"].astype(str).str.split("|", expand=True).astype(int)
        white_vals = np.arange(1, 70, dtype=np.int64)
        w_counts = np.zeros_like(white_vals, dtype=np.float64)
        # count occurrences
        flat_white = white.values.reshape(-1)
        for v in flat_white:
            if 1 <= int(v) <= 69:
                w_counts[int(v) - 1] += 1.0

        # Laplace smoothing
        w_counts += float(smoothing)
        w_probs = w_counts / w_counts.sum()

        # ----- Red balls (1..26) -----
        red_vals = np.arange(1, 27, dtype=np.int64)
        r_counts = np.zeros_like(red_vals, dtype=np.float64)
        for v in df["red_ball"].astype(int).values:
            if 1 <= int(v) <= 26:
                r_counts[int(v) - 1] += 1.0
        r_counts += float(smoothing)
        r_probs = r_counts / r_counts.sum()

        self.white_vals = white_vals
        self.red_vals = red_vals
        self._white_empirical = w_probs
        self._red_empirical = r_probs

        self.T_white_min = float(T_white_min)
        self.T_red_min = float(T_red_min)

    @staticmethod
    def _mix_probs(empirical: np.ndarray, *, T: float, max_T: float) -> np.ndarray:
        if max_T <= 0:
            raise ValueError("max_T must be > 0")
        alpha = float(T) / float(max_T)
        if alpha < 0:
            alpha = 0.0
        elif alpha > 1:
            alpha = 1.0
        u = np.full_like(empirical, 1.0 / empirical.size, dtype=np.float64)
        p = (1.0 - alpha) * empirical + alpha * u
        # numerical guard
        p = np.maximum(p, 0.0)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            # fallback to uniform
            return u
        return p / s

    @staticmethod
    def _rng_from(rng: Optional[np.random.Generator], seed: Optional[int]) -> np.random.Generator:
        if rng is not None:
            return rng
        return np.random.default_rng(seed)

    def generate_ticket_batch(
        self,
        n: int,
        *,
        max_T: float = 100.0,
        include_metadata: bool = True,
        # Optional overrides (if None, sample uniformly from [T_min, max_T])
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
        If ensure_unique=True, this function guarantees that returned tickets are unique with respect
        to (sorted whites, red). If `existing_tickets` is provided, uniqueness is enforced against
        that set as well.

        The function resamples only as many tickets as needed to reach `n` unique tickets. It does
        not attempt to "refund budget" because this API is count-based (n tickets).
        """
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        if n == 0:
            return []

        max_T = float(max_T)
        if max_T <= 0:
            raise ValueError("max_T must be > 0")

        rng_ = self._rng_from(rng, seed)

        # Seed uniqueness set
        seen: Set[Tuple[int, int, int, int, int, int]] = set()
        if existing_tickets is not None:
            for t in existing_tickets:
                seen.add(Ticket.from_any(t).as_tuple)

        out: List[Dict[str, Any]] = []
        rounds = 0

        # Helper: sample one ticket (with temps), return canonical + payload
        def _sample_one() -> Tuple[Ticket, Dict[str, Any]]:
            # Sample per-ticket temperatures, unless explicitly provided
            Tw = float(T_white) if T_white is not None else float(rng_.uniform(self.T_white_min, max_T))
            Tr = float(T_red) if T_red is not None else float(rng_.uniform(self.T_red_min, max_T))

            # Clip to [0, max_T]
            Tw = max(0.0, min(Tw, max_T))
            Tr = max(0.0, min(Tr, max_T))

            p_w = self._mix_probs(self._white_empirical, T=Tw, max_T=max_T)
            p_r = self._mix_probs(self._red_empirical, T=Tr, max_T=max_T)

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

            # Oversample a bit to reduce the number of rounds at modest n.
            # Keep bounded to avoid large spikes in memory/time for huge n.
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
