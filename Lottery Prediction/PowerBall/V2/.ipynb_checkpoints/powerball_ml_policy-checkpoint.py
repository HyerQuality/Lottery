from __future__ import annotations

import pandas as pd
import numpy as np
from powerball_backtester import PowerballBacktester
from powerball_ml_ticket_generator import PowerballMLTicketGenerator


def policy_search_on_val(
    ml: PowerballMLTicketGenerator,
    *,
    draw_csv: str,
    jackpot_csv: str,
    ticket_budget: int,
    temperatures=(0.7, 0.9, 1.0, 1.1, 1.3),
    ensemble_sizes=(3, 5, 9),
    seeds=range(20),               # 20 Monte Carlo runs per policy
    use_multiplier: bool = False,  # start without it (per your request)
):
    # Determine which draw indices correspond to ML validation targets.
    # X rows correspond to draws 0..N-2 predicting draws 1..N-1.
    assert ml.splits_ is not None
    n_train = len(ml.splits_.X_train)
    n_val = len(ml.splits_.X_val)

    # Validation targets correspond to draw indices: (n_train+1) .. (n_train+n_val)
    val_start = n_train + 1
    val_end = n_train + n_val

    results = []

    for M in ensemble_sizes:
        for T in temperatures:
            vals = []
            for s in seeds:
                gen = MLBacktesterGenerator(ml, temperature=T, ensemble_size=M, seed=int(s))
                bt = PowerballBacktester(
                    draw_csv=draw_csv,
                    jackpot_csv=jackpot_csv,
                    generator=gen,
                    ticket_budget=ticket_budget,
                    use_multiplier=use_multiplier,  # multiplier applied per draw by the backtester when True
                    reinvest_percent=0.0,
                    store_temperatures=True,
                    seed=int(s),
                )
                pnl = bt.run(seed=int(s))

                # Score ONLY on the validation draw window using incremental profit
                pnl_window = pnl.iloc[val_start:val_end + 1]
                score = float((pnl_window["draw_payout"] - pnl_window["spend"]).sum())
                vals.append(score)

            results.append(
                {
                    "ensemble_size": int(M),
                    "temperature": float(T),
                    "mc_runs": int(len(list(seeds))),
                    "score_mean": float(np.mean(vals)),
                    "score_median": float(np.median(vals)),
                    "score_p10": float(np.quantile(vals, 0.10)),
                    "score_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            )

    df = pd.DataFrame(results).sort_values(["score_median", "score_p10"], ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    return best, df


class MLBacktesterGenerator:
    """
    Adapter that presents generate_ticket_batch(...) to PowerballBacktester.

    Key behavior:
    - Advances one draw per outer cycle (existing_tickets is None for the first call in a draw).
    - Uses history up to i-1 to predict for draw i via features_for_next_draw(...).
    - Enforces uniqueness across multiplier/non-multiplier pools using existing_tickets.
    - Uses only the top-M ensemble members per head (M set at init).
    """
    def __init__(self, ml: PowerballMLTicketGenerator, *, temperature: float, ensemble_size: int, seed: int = 123):
        self.ml = ml
        self.temperature = float(temperature)
        self.M = int(ensemble_size)
        self.seed = int(seed)
        self._draw_index = -1

    def _trim(self, pipes):
        return pipes[: self.M]

    def _ensemble_proba_trimmed(self, head: str, X_row: pd.DataFrame, *, n_classes: int, class_min: int) -> np.ndarray:
        if head == "R":
            ens = self._trim(self.ml.red_ensemble_)
        else:
            ens = self._trim(self.ml.white_ensembles_[head])
        # Public wrapper (keeps correct class alignment)
        return self.ml.ensemble_proba(ens, X_row, n_classes=n_classes, class_min=class_min)

    def generate_ticket_batch(
        self,
        n: int,
        max_T: float = 100.0,          # ignored (compat)
        include_metadata: bool = True,
        existing_tickets=None,
        ensure_unique: bool = True,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        rng = rng or np.random.default_rng(self.seed if seed is None else int(seed))

        # Backtester calls multiplier pool first (existing_tickets=None), then base pool (existing_tickets=mult_pool)
        if existing_tickets is None:
            self._draw_index += 1

        i = self._draw_index

        # For draw 0 we don't have history; fallback uniform
        if i <= 0:
            pW = {f"W{k}": np.ones(69) / 69.0 for k in range(1, 6)}
            pR = np.ones(26) / 26.0
        else:
            # Build X row for draw i using history up to i-1
            df_hist = self.ml.draws.iloc[:i].copy()
            X_row = self.ml.features_for_next_draw(df_hist).to_frame().T

            pW = {f"W{k}": self._ensemble_proba_trimmed(f"W{k}", X_row, n_classes=69, class_min=1) for k in range(1, 6)}
            pR = self._ensemble_proba_trimmed("R", X_row, n_classes=26, class_min=1)

        # uniqueness across pools
        seen = set()
        if existing_tickets is not None:
            for t in existing_tickets:
                w = list(map(int, t.get("white_balls", []))) or [int(t[f"white_{k}"]) for k in range(1, 6)]
                w.sort()
                seen.add((*w, int(t["red_ball"])))

        out = []
        attempts = 0
        while len(out) < int(n):
            attempts += 1
            if attempts > 200_000:
                raise RuntimeError("Unable to sample enough unique tickets for this draw.")

            whites = self.ml.sample_whites_from_head_probas(pW, temperature=self.temperature, rng=rng)
            red = int(rng.choice(np.arange(1, 27), p=self.ml.apply_temperature(pR.copy(), self.temperature)))

            key = (*whites, red)
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
                out.append(
                    {
                        "white_1": whites[0],
                        "white_2": whites[1],
                        "white_3": whites[2],
                        "white_4": whites[3],
                        "white_5": whites[4],
                        "red_ball": red,
                    }
                )
        return out