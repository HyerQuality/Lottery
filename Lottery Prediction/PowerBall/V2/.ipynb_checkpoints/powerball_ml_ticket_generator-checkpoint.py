from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import joblib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class SplitData:
    X_train: pd.DataFrame
    yW_train: np.ndarray  # (n_train, 5) next-draw whites (sorted)
    yR_train: np.ndarray  # (n_train,) next-draw red
    X_val: pd.DataFrame
    yW_val: np.ndarray
    yR_val: np.ndarray
    X_test: pd.DataFrame
    yW_test: np.ndarray
    yR_test: np.ndarray


class PowerballMLTicketGenerator:
    """
    ML-driven ticket generator (revised skeleton).

    Key design choices implemented:
      - Predict next draw (X_t -> y_{t+1})
      - 5 exchangeable white-ball heads (multiclass 1..69 each) + 1 red head (1..26)
      - Training augmentation: duplicate rows with random permutations of the 5 whites so heads
        do not learn sorted-position artifacts
      - Chronological train/val/test split
      - HistGradientBoostingClassifier as default model family
      - QuantileTransformer off by default (can enable for non-tree models later)
      - DOW + month treated as categorical via OneHotEncoder in the pipeline
      - Optional hyperparameter optimization + MC ensemble of slightly different models
      - Ticket generation: head-order randomization + sequential sampling w/out replacement + temperature
      - Evaluation: log loss + multiclass Brier score (plus match counts if desired)
    """

    # -----------------------------
    # Construction
    # -----------------------------
    def __init__(
        self,
        draw_data: Union[str, pd.DataFrame],
        *,
        lag_n: int = 10,
        rolling_windows: Tuple[int, ...] = (5, 10, 20),
        val_size: float = 0.15,
        test_size: float = 0.15,
        augment_permutations: int = 8,
        # preprocessing
        use_quantile: bool = False,  # keep default OFF for boosting/forests
        # modeling
        base_model: Optional[BaseEstimator] = None,
        # tuning / ensemble
        enable_tuning: bool = True,
        tuning_n_iter: int = 30,
        tuning_cv_splits: int = 5,
        n_jobs: int = 4,
        mc_ensemble_size: int = 7,
        mc_strategy: str = "repeated_random_search",  # or "parameter_sampler"
        param_distributions: Optional[Dict[str, Any]] = None,
        # reproducibility
        seed: int = 123,
        verbose: bool = False,
    ) -> None:
        self.lag_n = int(lag_n)
        self.rolling_windows = tuple(int(w) for w in rolling_windows)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.augment_permutations = int(augment_permutations)
        self.use_quantile = bool(use_quantile)

        self.enable_tuning = bool(enable_tuning)
        self.tuning_n_iter = int(tuning_n_iter)
        self.tuning_cv_splits = int(tuning_cv_splits)
        self.n_jobs = int(n_jobs)
        self.mc_ensemble_size = int(mc_ensemble_size)
        self.mc_strategy = str(mc_strategy)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self._validate_config()

        self.draws_raw = self._load(draw_data)
        self.draws = self._canonicalize_and_parse(self.draws_raw)

        self.base_model = base_model or HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.0,
            max_bins=255,
            random_state=self.seed,

            # Anti-overfit defaults
            early_stopping=False,
            validation_fraction=0.50,
            n_iter_no_change=20,
            tol=1e-7,
            max_iter=500,
        )

        self.param_distributions = param_distributions or self._default_param_distributions()

        # fitted artifacts
        self.feature_columns_: Optional[List[str]] = None
        self.numeric_cols_: Optional[List[str]] = None
        self.categorical_cols_: Optional[List[str]] = None
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.splits_: Optional[SplitData] = None

        # models: ensembles of estimators per head
        self.white_ensembles_: Dict[str, List[Pipeline]] = {}  # keys: "W1".."W5"
        self.red_ensemble_: List[Pipeline] = []

    # -----------------------------
    # Public API
    # -----------------------------
    def _validate_config(self) -> None:
        """Validate construction-time configuration.

        Kept intentionally lightweight to avoid altering behavior, while failing fast on
        clearly invalid split fractions.
        """
        if self.val_size < 0.0 or self.test_size < 0.0 or (self.val_size + self.test_size) >= 1.0:
            raise ValueError(
                f"Invalid val/test sizes: val_size={self.val_size}, test_size={self.test_size}. "
                "Require val_size>=0, test_size>=0, and val_size+test_size<1."
            )

    # -----------------------------
    # Determinism + parsing helpers
    # -----------------------------
    @staticmethod
    def _stable_head_offset(head_name: str) -> int:
        """Deterministic integer offset for per-head RNG seeding.

        Avoids Python's randomized hash(), ensuring reproducible ensembles across processes.
        """
        digest = hashlib.sha256(str(head_name).encode("utf-8")).digest()
        # 32-bit offset is sufficient and stable
        return int.from_bytes(digest[:4], "little", signed=False)

    @staticmethod
    def _parse_whites(series: pd.Series) -> pd.DataFrame:
        """Parse a column of white-ball strings into a (n,5) int DataFrame.

        Accepts common delimiters ("|", ",", spaces) by extracting digit groups.
        Raises ValueError with context if any row fails to yield 5 valid balls.
        """
        found = series.astype(str).str.findall(r"\d+")
        rows = []
        bad_idx = []
        for i, vals in enumerate(found.tolist()):
            nums = [int(v) for v in vals][:5]
            if len(nums) != 5:
                bad_idx.append(i)
                rows.append([np.nan] * 5)
                continue
            rows.append(nums)

        whites = pd.DataFrame(rows, columns=[f"white_{k}" for k in range(1, 6)])

        if bad_idx:
            examples = series.iloc[bad_idx[:5]].astype(str).tolist()
            raise ValueError(
                f"Failed to parse 5 white balls from {len(bad_idx)} row(s). "
                f"Examples: {examples}"
            )

        arr = whites.to_numpy(dtype=int)

        # Basic validation: range and uniqueness per draw
        if (arr < 1).any() or (arr > 69).any():
            raise ValueError("White balls must be integers in [1, 69].")
        if any(len(set(row)) != 5 for row in arr):
            raise ValueError("White balls must be unique within each draw.")

        return whites.astype(int)

    def build(self) -> "PowerballMLTicketGenerator":
        X, yW, yR = self._make_supervised_frame(self.draws)
        self.splits_ = self._split_chronological(X, yW, yR)

        self.feature_columns_ = list(self.splits_.X_train.columns)
        self.numeric_cols_, self.categorical_cols_ = self._infer_column_types(self.splits_.X_train)
        self.preprocessor_ = self._build_preprocessor(self.numeric_cols_, self.categorical_cols_)

        return self

    def fit(self) -> "PowerballMLTicketGenerator":
        if self.splits_ is None or self.preprocessor_ is None:
            self.build()

        if self.splits_ is None:
            raise RuntimeError("Internal error: splits_ is not initialized. Call build() first.")
        if self.preprocessor_ is None:
            raise RuntimeError("Internal error: preprocessor_ is not initialized. Call build() first.")

        # --- build augmented training data for white heads
        XW_train, yW_heads_train = self._augment_whites(
            self.splits_.X_train,
            self.splits_.yW_train,  # (n,5)
            n_permutations=self.augment_permutations,
            rng=np.random.default_rng(self.seed),
        )

        # --- fit ensembles for W1..W5
        self.white_ensembles_.clear()
        for i in range(1, 6):
            head = f"W{i}"
            y_head = yW_heads_train[head]  # shape (n_aug,)
            self.white_ensembles_[head] = self._fit_mc_ensemble(
                X_train=XW_train,
                y_train=y_head,
                X_val=self.splits_.X_val,
                y_val=self.splits_.yW_val[:, i - 1],
                head_name=head,
                n_classes=69,
                class_min=1,
            )

        # --- fit ensemble for red head (no augmentation needed)
        self.red_ensemble_ = self._fit_mc_ensemble(
            X_train=self.splits_.X_train,
            y_train=self.splits_.yR_train,
            X_val=self.splits_.X_val,
            y_val=self.splits_.yR_val,
            head_name="R",
            n_classes=26,
            class_min=1,
        )

        if self.verbose:
            self.print_eval(split="val")

        return self

    def predict_next_distribution(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Returns:
          p_whites_by_head: dict {"W1":(69,),...,"W5":(69,)} probability vectors over 1..69
          p_red: (26,) probability vector over 1..26
        """
        X_next = self._features_for_next_draw(self.draws).to_frame().T

        p_whites: Dict[str, np.ndarray] = {}
        for i in range(1, 6):
            head = f"W{i}"
            p_whites[head] = self._ensemble_proba(
                self.white_ensembles_.get(head, []),
                X_next,
                n_classes=69,
                class_min=1,
            )

        p_red = self._ensemble_proba(self.red_ensemble_, X_next, n_classes=26, class_min=1)
        return p_whites, p_red

    # -----------------------------
    # Public helper API (for adapters/policies/backtesting)
    # -----------------------------
    def features_for_next_draw(self, df_hist: pd.DataFrame) -> pd.Series:
        """Public wrapper: build the feature row for the next draw from historical draws."""
        return self._features_for_next_draw(df_hist)

    def ensemble_proba(
        self,
        ensemble: Sequence[Any],
        X_row: pd.DataFrame,
        *,
        n_classes: int,
        class_min: int,
    ) -> np.ndarray:
        """Public wrapper: ensemble-averaged class probability with full class alignment."""
        return self._ensemble_proba(ensemble, X_row, n_classes=n_classes, class_min=class_min)

    def apply_temperature(self, p: np.ndarray, temperature: float) -> np.ndarray:
        """Public wrapper: apply sampling temperature to a probability vector."""
        return self._apply_temperature(p, temperature)

    def sample_whites_from_head_probas(
        self,
        p_whites_by_head: Dict[str, np.ndarray],
        *,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        randomize_head_order: bool = True,
    ) -> List[int]:
        """Sample 5 unique white balls from per-head probability vectors over 1..69.

        Uses sequential sampling without replacement (mask + renormalize) and optional temperature.
        """
        rng = rng or np.random.default_rng(self.seed)

        heads = list(p_whites_by_head.keys())
        if randomize_head_order:
            rng.shuffle(heads)
        else:
            heads = sorted(heads)

        chosen: List[int] = []
        chosen_set = set()

        for h in heads:
            p = np.asarray(p_whites_by_head[h], dtype=np.float64).copy()
            if p.shape[0] != 69:
                raise ValueError(f"Expected 69-class white distribution for {h}, got shape {p.shape}")

            # mask already-chosen whites (index is ball-1)
            for w in chosen_set:
                p[w - 1] = 0.0

            p = self._renorm(p)
            p = self.apply_temperature(p, temperature)
            w = int(rng.choice(np.arange(1, 70), p=p))
            chosen.append(w)
            chosen_set.add(w)

        chosen.sort()
        return chosen

    def generate_tickets(
        self,
        n: int,
        *,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
        as_flat: bool = True,
    ) -> List[Dict[str, int]]:
        """
        Ticket generation:
          - head-order randomization
          - sequential sampling without replacement (mask + renormalize)
          - optional temperature on each step
        """
        if not self.white_ensembles_ or not self.red_ensemble_:
            raise RuntimeError("Call fit() before generate_tickets().")

        rng = rng or np.random.default_rng(self.seed)
        p_whites_by_head, p_red = self.predict_next_distribution()

        tickets: List[Dict[str, int]] = []
        seen = set()
        attempts = 0

        while len(tickets) < int(n):
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError("Unable to sample enough unique tickets; relax uniqueness or adjust temperature.")
            chosen_whites = self.sample_whites_from_head_probas(
                p_whites_by_head,
                temperature=temperature,
                rng=rng,
                randomize_head_order=True,
            )

            # sample_whites_from_head_probas returns sorted whites
            red = int(rng.choice(np.arange(1, 27), p=self._apply_temperature(p_red.copy(), temperature)))

            key = (*chosen_whites, red)
            if key in seen:
                continue
            seen.add(key)

            if as_flat:
                tickets.append(
                    {
                        "white_1": chosen_whites[0],
                        "white_2": chosen_whites[1],
                        "white_3": chosen_whites[2],
                        "white_4": chosen_whites[3],
                        "white_5": chosen_whites[4],
                        "red_ball": red,
                    }
                )
            else:
                tickets.append({"white_balls": chosen_whites, "red_ball": red})

        return tickets

    def evaluate(
        self,
        *,
        split: str = "val",
    ) -> Dict[str, float]:
        """
        Returns mean log loss and multiclass Brier across heads, plus red head metrics.
        """
        if self.splits_ is None:
            raise RuntimeError("Call build()/fit() first.")

        X, yW, yR = self._get_split(split)

        out: Dict[str, float] = {}
        ll_whites = []
        bs_whites = []

        for i in range(1, 6):
            head = f"W{i}"
            p = self._ensemble_proba(self.white_ensembles_.get(head, []), X, n_classes=69, class_min=1)
            y = yW[:, i - 1].astype(int)
            ll_whites.append(self._log_loss_fixed(y, p, n_classes=69, class_min=1))
            bs_whites.append(self._brier_multiclass_fixed(y, p, n_classes=69, class_min=1))

        pR = self._ensemble_proba(self.red_ensemble_, X, n_classes=26, class_min=1)
        out["white_log_loss_mean"] = float(np.mean(ll_whites))
        out["white_brier_mean"] = float(np.mean(bs_whites))
        out["red_log_loss"] = float(self._log_loss_fixed(yR.astype(int), pR, n_classes=26, class_min=1))
        out["red_brier"] = float(self._brier_multiclass_fixed(yR.astype(int), pR, n_classes=26, class_min=1))
        return out

    def print_eval(self, *, split: str = "val") -> None:
        m = self.evaluate(split=split)
        print(f"[{split}] white log_loss(mean): {m['white_log_loss_mean']:.4f} | white brier(mean): {m['white_brier_mean']:.4f}")
        print(f"[{split}] red   log_loss:       {m['red_log_loss']:.4f} | red   brier:       {m['red_brier']:.4f}")

    # -----------------------------
    # Persistence
    # -----------------------------
    STATE_SCHEMA_VERSION = 1

    def to_state_dict(self, *, include_draws: bool = False) -> Dict[str, Any]:
        """Return a versioned, pickle-friendly state dict for persistence.

        Notes:
          - Stores fitted sklearn objects (preprocessor + fitted pipelines). It does NOT
            store the entire instance via pickle, which reduces fragility across refactors.
          - If include_draws=False, you must pass draw_data on load (or rebuild draws separately).
        """
        state: Dict[str, Any] = {
            "schema_version": self.STATE_SCHEMA_VERSION,
            "class_name": self.__class__.__name__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": {
                "lag_n": self.lag_n,
                "rolling_windows": self.rolling_windows,
                "val_size": self.val_size,
                "test_size": self.test_size,
                "augment_permutations": self.augment_permutations,
                "use_quantile": self.use_quantile,
                "enable_tuning": self.enable_tuning,
                "tuning_n_iter": self.tuning_n_iter,
                "tuning_cv_splits": self.tuning_cv_splits,
                "mc_ensemble_size": self.mc_ensemble_size,
                "mc_strategy": self.mc_strategy,
                "seed": self.seed,
                "verbose": self.verbose,
            },
            "feature_columns_": self.feature_columns_,
            "numeric_cols_": self.numeric_cols_,
            "categorical_cols_": self.categorical_cols_,
            "preprocessor_": self.preprocessor_,
            "param_distributions": self.param_distributions,
            "base_model": self.base_model,
            "white_ensembles_": self.white_ensembles_,
            "red_ensemble_": self.red_ensemble_,
        }

        # Helpful fingerprinting (not required for loading)
        if hasattr(self, "draws") and isinstance(self.draws, pd.DataFrame) and not self.draws.empty:
            fp_src = (
                str(self.draws.shape)
                + str(self.draws["date"].min())
                + str(self.draws["date"].max())
            ).encode("utf-8", errors="ignore")
            state["draws_fingerprint"] = hashlib.sha256(fp_src).hexdigest()

        if include_draws:
            state["draws"] = self.draws.copy()

        return state

    @classmethod
    def from_state_dict(
        cls,
        state: Dict[str, Any],
        *,
        draw_data: Optional[Union[str, pd.DataFrame]] = None,
    ) -> "PowerballMLTicketGenerator":
        """Rehydrate an instance from a state dict."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")

        schema_version = state.get("schema_version")
        if schema_version != cls.STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version}. Expected {cls.STATE_SCHEMA_VERSION}."
            )

        cfg = state.get("config", {})

        if draw_data is None:
            if "draws" not in state:
                raise ValueError(
                    "No draw_data provided and state dict does not include draws. "
                    "Load with draw_data=... or save with include_draws=True."
                )
            draw_data = state["draws"]

        obj = cls(
            draw_data=draw_data,
            lag_n=int(cfg.get("lag_n", 10)),
            rolling_windows=tuple(cfg.get("rolling_windows", (5, 10, 20))),
            val_size=float(cfg.get("val_size", 0.15)),
            test_size=float(cfg.get("test_size", 0.15)),
            augment_permutations=int(cfg.get("augment_permutations", 8)),
            use_quantile=bool(cfg.get("use_quantile", False)),
            base_model=state.get("base_model", None),
            enable_tuning=bool(cfg.get("enable_tuning", True)),
            tuning_n_iter=int(cfg.get("tuning_n_iter", 30)),
            tuning_cv_splits=int(cfg.get("tuning_cv_splits", 5)),
            mc_ensemble_size=int(cfg.get("mc_ensemble_size", 7)),
            mc_strategy=str(cfg.get("mc_strategy", "repeated_random_search")),
            param_distributions=state.get("param_distributions", None),
            seed=int(cfg.get("seed", 123)),
            verbose=bool(cfg.get("verbose", False)),
        )

        obj.feature_columns_ = state.get("feature_columns_")
        obj.numeric_cols_ = state.get("numeric_cols_")
        obj.categorical_cols_ = state.get("categorical_cols_")
        obj.preprocessor_ = state.get("preprocessor_")
        obj.white_ensembles_ = state.get("white_ensembles_", {}) or {}
        obj.red_ensemble_ = state.get("red_ensemble_", []) or []
        obj.splits_ = None

        return obj

    def save_state(self, path: str, *, include_draws: bool = False, compress: int = 3) -> None:
        """Save a state dict to disk via joblib."""
        state = self.to_state_dict(include_draws=include_draws)
        joblib.dump(state, path, compress=compress)

    @classmethod
    def load_state(
        cls,
        path: str,
        *,
        draw_data: Optional[Union[str, pd.DataFrame]] = None,
    ) -> "PowerballMLTicketGenerator":
        """Load a state dict from disk and rehydrate an instance."""
        state = joblib.load(path)
        return cls.from_state_dict(state, draw_data=draw_data)

    # -----------------------------
    # Loading + parsing
    # -----------------------------
    def _load(self, draw_data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(draw_data, str):
            return pd.read_csv(draw_data)
        return draw_data.copy()

    def _canonicalize_and_parse(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        cols_norm = {c: "".join(ch for ch in str(c).strip().lower() if ch.isalnum()) for c in out.columns}
        inv = {v: k for k, v in cols_norm.items()}

        def pick(*names: str) -> str:
            for n in names:
                nn = "".join(ch for ch in n.strip().lower() if ch.isalnum())
                if nn in inv:
                    return inv[nn]
            raise ValueError(f"Missing required column; tried: {names}. Found: {list(out.columns)}")

        date_col = pick("date", "draw_date", "drawdate")
        white_col = pick("white_balls", "whiteballs", "white_numbers", "whites")
        red_col = pick("red_ball", "redball", "powerball", "pb")

        if date_col != "date":
            out.rename(columns={date_col: "date"}, inplace=True)
        if white_col != "white_balls":
            out.rename(columns={white_col: "white_balls"}, inplace=True)
        if red_col != "red_ball":
            out.rename(columns={red_col: "red_ball"}, inplace=True)

        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        whites = self._parse_whites(out["white_balls"])
        whites.columns = [f"white_{i}" for i in range(1, 6)]

        out = pd.concat([out[["date", "red_ball"]], whites], axis=1)
        out["red_ball"] = pd.to_numeric(out["red_ball"], errors="coerce").astype(int)

        # canonicalize: whites sorted per draw
        w = out[[f"white_{i}" for i in range(1, 6)]].to_numpy(dtype=int)
        w.sort(axis=1)
        for i in range(5):
            out[f"white_{i+1}"] = w[:, i]

        return out

    # -----------------------------
    # Feature engineering + supervised framing
    # -----------------------------
    def _make_supervised_frame(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        feats = self._engineer_features(df)

        # next-draw labels
        next_df = df.shift(-1)

        # IMPORTANT: slice off the trailing NaN row BEFORE casting to int
        yR = next_df["red_ball"].iloc[:-1].to_numpy(dtype=int)
        yW = next_df[[f"white_{i}" for i in range(1, 6)]].iloc[:-1].to_numpy(dtype=int)
        # ensure sorted labels
        yW.sort(axis=1)

        X = feats.iloc[:-1].reset_index(drop=True)
        return X, yW, yR

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Expect df has: date, red_ball, white_1..white_5 (whites already sorted)
        wcols = [f"white_{i}" for i in range(1, 6)]
        idx = df.index
    
        w = df[wcols].to_numpy(dtype=np.float64)         # (n,5)
        red = df["red_ball"].to_numpy(dtype=np.float64)  # (n,)
    
        feats: Dict[str, Any] = {}
    
        # ---- base stats
        w_sum = w.sum(axis=1)
        w_mean = w.mean(axis=1)
        w_std = w.std(axis=1)
        w_min = w.min(axis=1)
        w_max = w.max(axis=1)
        w_range = w_max - w_min
    
        feats["w_sum"] = w_sum
        feats["w_mean"] = w_mean
        feats["w_std"] = w_std
        feats["w_min"] = w_min
        feats["w_max"] = w_max
        feats["w_range"] = w_range
    
        # ---- gaps (whites sorted)
        gaps = np.diff(w, axis=1)  # (n,4)
        for i in range(1, 5):
            feats[f"gap_{i}"] = gaps[:, i - 1]
        feats["gap_mean"] = gaps.mean(axis=1)
        feats["gap_std"] = gaps.std(axis=1)
    
        # ---- parity / high-low counts
        feats["w_odd_cnt"] = (w.astype(np.int64) % 2).sum(axis=1).astype(np.float64)
        feats["w_high_cnt"] = (w.astype(np.int64) >= 35).sum(axis=1).astype(np.float64)
    
        # ---- ratios with red
        eps = 1e-9
        feats["r_over_w_mean"] = red / (w_mean + eps)
        feats["r_over_w_sum"] = red / (w_sum + eps)
        feats["r_plus_w_sum"] = red + w_sum
    
        # ---- light polynomial terms (degree 2 only)
        # raw balls
        for i, c in enumerate(wcols):
            feats[f"{c}^2"] = w[:, i] ** 2
        feats["red_ball^2"] = red ** 2
        feats["w_sum^2"] = w_sum ** 2
        feats["w_mean^2"] = w_mean ** 2
        feats["w_range^2"] = w_range ** 2
    
        # ---- rolling stats (use pandas rolling once per series; store results in dict)
        s_w_sum = pd.Series(w_sum, index=idx)
        s_red = pd.Series(red, index=idx)
    
        for win in self.rolling_windows:
            mp = max(2, win // 3)
            feats[f"w_sum_roll_mean_{win}"] = s_w_sum.rolling(win, min_periods=mp).mean().to_numpy()
            feats[f"w_sum_roll_std_{win}"] = s_w_sum.rolling(win, min_periods=mp).std(ddof=0).to_numpy()
            feats[f"red_roll_mean_{win}"] = s_red.rolling(win, min_periods=mp).mean().to_numpy()
    
        # ---- lag features (shift once per base series)
        base_series = {
            **{c: pd.Series(w[:, i], index=idx) for i, c in enumerate(wcols)},
            "red_ball": s_red,
            "w_sum": s_w_sum,
            "w_mean": pd.Series(w_mean, index=idx),
            "w_std": pd.Series(w_std, index=idx),
            "w_range": pd.Series(w_range, index=idx),
            "w_odd_cnt": pd.Series(feats["w_odd_cnt"], index=idx),
            "w_high_cnt": pd.Series(feats["w_high_cnt"], index=idx),
        }
    
        for k in range(1, self.lag_n + 1):
            for name, ser in base_series.items():
                feats[f"{name}_lag{k}"] = ser.shift(k).to_numpy()
    
        # ---- categorical calendar features (kept as float64 for pipeline)
        feats["dow"] = df["date"].dt.dayofweek.astype("float64")
        feats["month"] = df["date"].dt.month.astype("float64")
    
        # Single materialization: no fragmentation
        return pd.DataFrame(feats, index=idx)

    def _features_for_next_draw(self, df: pd.DataFrame) -> pd.Series:
        feats = self._engineer_features(df)
        last = feats.iloc[-1]
        if self.feature_columns_ is not None:
            last = last.reindex(self.feature_columns_)
        return last

    # -----------------------------
    # Splitting
    # -----------------------------
    def _split_chronological(self, X: pd.DataFrame, yW: np.ndarray, yR: np.ndarray) -> SplitData:
        n = len(X)
        n_test = max(1, int(round(n * self.test_size)))
        n_val = max(1, int(round(n * self.val_size)))
        n_train = n - n_val - n_test
        if n_train <= 0:
            raise ValueError("train/val/test sizes leave no training data.")

        idx_train = np.arange(0, n_train)
        idx_val = np.arange(n_train, n_train + n_val)
        idx_test = np.arange(n_train + n_val, n)

        return SplitData(
            X_train=X.iloc[idx_train].reset_index(drop=True),
            yW_train=yW[idx_train],
            yR_train=yR[idx_train],
            X_val=X.iloc[idx_val].reset_index(drop=True),
            yW_val=yW[idx_val],
            yR_val=yR[idx_val],
            X_test=X.iloc[idx_test].reset_index(drop=True),
            yW_test=yW[idx_test],
            yR_test=yR[idx_test],
        )

    def _get_split(self, split: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if self.splits_ is None:
            raise RuntimeError("Internal error: splits_ is not initialized. Call build() first.")
        s = split.lower().strip()
        if s == "train":
            return self.splits_.X_train, self.splits_.yW_train, self.splits_.yR_train
        if s == "val":
            return self.splits_.X_val, self.splits_.yW_val, self.splits_.yR_val
        if s == "test":
            return self.splits_.X_test, self.splits_.yW_test, self.splits_.yR_test
        raise ValueError("split must be one of: train, val, test")

    # -----------------------------
    # Augmentation for exchangeable white heads
    # -----------------------------
    def _augment_whites(
        self,
        X: pd.DataFrame,
        yW: np.ndarray,  # (n,5)
        *,
        n_permutations: int,
        rng: np.random.Generator,
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        n = len(X)
        K = max(1, int(n_permutations))

        # Duplicate X K times
        X_aug = pd.concat([X] * K, axis=0, ignore_index=True)

        # For each replicate, permute the 5 labels per row and assign position i -> Wi
        y_heads: Dict[str, np.ndarray] = {f"W{i}": np.empty(n * K, dtype=int) for i in range(1, 6)}

        for k in range(K):
            start = k * n
            end = (k + 1) * n
            block = yW.copy()
            for i in range(n):
                rng.shuffle(block[i])  # in-place permutation of 5 whites
            for i in range(5):
                y_heads[f"W{i+1}"][start:end] = block[:, i]

        return X_aug, y_heads

    # -----------------------------
    # Preprocessing (Pipeline-ready)
    # -----------------------------
    def _infer_column_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        cat = [c for c in X.columns if c in ("dow", "month")]
        num = [c for c in X.columns if c not in cat]
        return num, cat

    def _build_preprocessor(self, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        num_steps: List[Tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
        if self.use_quantile:
            num_steps.append(
                ("qt", QuantileTransformer(output_distribution="normal", random_state=self.seed))
            )
        num_pipe = Pipeline(steps=num_steps)

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def _make_pipeline(self, *, random_state: int) -> Pipeline:
        if self.preprocessor_ is None:
            raise RuntimeError("Internal error: preprocessor_ is not initialized. Call build() first.")
        mdl = clone(self.base_model)
        if hasattr(mdl, "random_state"):
            setattr(mdl, "random_state", int(random_state))
        return Pipeline(steps=[("prep", self.preprocessor_), ("model", mdl)])

    # -----------------------------
    # Tuning + MC ensemble
    # -----------------------------
    def _default_param_distributions(self) -> Dict[str, Any]:
        # Narrow band around reasonable defaults; adjust as needed.
        # Prefix "model__" because we tune inside Pipeline.
        return {
            "model__learning_rate": np.linspace(0.02, 0.12, 10),
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [10, 20, 40, 80],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
            "model__max_depth": [None, 3, 5],
        }

    def _fit_mc_ensemble(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        head_name: str,
        n_classes: int,
        class_min: int,
    ) -> List[Pipeline]:
        """
        Fit an MC ensemble for one head and optionally tune.

        Overfitting control:
          - Chronological holdout: X_val/y_val are NOT used in CV; only for selecting
            among ensemble members (and later policy tuning).
          - CV happens only on training via TimeSeriesSplit.
        """
        M = max(1, int(self.mc_ensemble_size))
        rng = np.random.default_rng(self.seed + self._stable_head_offset(head_name))

        if not self.enable_tuning:
            return [self._fit_one(X_train, y_train, random_state=int(rng.integers(1, 1_000_000))) for _ in range(M)]

        if self.mc_strategy not in ("repeated_random_search", "parameter_sampler"):
            raise ValueError("mc_strategy must be 'repeated_random_search' or 'parameter_sampler'")

        # Build candidate models
        candidates: List[Pipeline] = []

        if self.mc_strategy == "repeated_random_search":
            for m in range(M):
                rs = int(rng.integers(1, 1_000_000))
                best = self._fit_via_random_search(
                    X_train, y_train, random_state=rs, n_classes=n_classes, class_min=class_min
                )
                candidates.append(best)
        else:
            # parameter sampler: sample M hyperparam sets and fit directly
            param_list = list(
                ParameterSampler(self.param_distributions, n_iter=max(M, self.tuning_n_iter), random_state=int(rng.integers(1, 1_000_000)))
            )
            for j in range(M):
                rs = int(rng.integers(1, 1_000_000))
                pipe = self._make_pipeline(random_state=rs)
                pipe.set_params(**param_list[j])
                pipe.fit(X_train, y_train)
                candidates.append(pipe)

        # Score candidates on validation (log loss) and keep all (or top-K if desired later)
        scored: List[Tuple[float, Pipeline]] = []
        for pipe in candidates:
            p = self._proba_fixed(pipe, X_val, n_classes=n_classes, class_min=class_min)
            ll = self._log_loss_fixed(y_val.astype(int), p, n_classes=n_classes, class_min=class_min)
            scored.append((ll, pipe))

        scored.sort(key=lambda t: t[0])
        # Keep all M (already size M); sorted best->worst
        return [p for _, p in scored]

    def _fit_one(self, X: pd.DataFrame, y: np.ndarray, *, random_state: int) -> Pipeline:
        pipe = self._make_pipeline(random_state=random_state)
        pipe.fit(X, y)
        return pipe

    def _fit_via_random_search(self, X, y, *, random_state: int, n_classes: int, class_min: int) -> Pipeline:
        pipe = self._make_pipeline(random_state=random_state)
    
        def neg_log_loss_fixed(estimator, X_fold, y_fold):
            p = self._proba_fixed(estimator, X_fold, n_classes=n_classes, class_min=class_min)
            return -self._log_loss_fixed(y_fold.astype(int), p, n_classes=n_classes, class_min=class_min)
    
        cv = TimeSeriesSplit(n_splits=max(2, int(self.tuning_cv_splits)))
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=self.param_distributions,
            n_iter=max(1, int(self.tuning_n_iter)),
            scoring=neg_log_loss_fixed,   # <- avoids missing classes errors by assigning epsilon prob to missing classes
            cv=cv,
            refit=True,
            random_state=int(random_state),
            n_jobs=self.n_jobs,
            verbose=2,
            error_score="raise",          # optional while verifying
        )
        search.fit(X, y)
        return search.best_estimator_

    # -----------------------------
    # Ensemble probability utilities
    # -----------------------------
    def _ensemble_proba(
        self,
        ensemble: Sequence[Pipeline],
        X: pd.DataFrame,
        *,
        n_classes: int,
        class_min: int,
    ) -> np.ndarray:
        if len(ensemble) == 0:
            raise RuntimeError("Ensemble is empty; did fit() run successfully?")

        ps = [self._proba_fixed(m, X, n_classes=n_classes, class_min=class_min) for m in ensemble]
        p = np.mean(ps, axis=0)
        p = np.clip(p, 1e-12, None)
        p = p / p.sum(axis=1, keepdims=True)
        # If X is 1-row, return 1D vector
        return p[0] if p.shape[0] == 1 else p

    def _proba_fixed(self, pipe: Pipeline, X: pd.DataFrame, *, n_classes: int, class_min: int) -> np.ndarray:
        """
        Returns a dense (n_samples, n_classes) array aligned to class labels {class_min..class_min+n_classes-1}.
        Handles missing classes by smoothing.
        """
        model = pipe.named_steps["model"]
        proba = pipe.predict_proba(X)  # may be (n, Kpresent)
        classes = getattr(model, "classes_", None)
        if classes is None:
            # Should not happen for sklearn classifiers with predict_proba
            raise RuntimeError("Model does not expose classes_ for probability alignment.")

        out = np.full((X.shape[0], n_classes), 1e-12, dtype=np.float64)
        for j, cls in enumerate(classes):
            c = int(cls)
            idx = c - int(class_min)
            if 0 <= idx < n_classes:
                out[:, idx] = proba[:, j].astype(np.float64)

        out = out / out.sum(axis=1, keepdims=True)
        return out

    # -----------------------------
    # Metrics
    # -----------------------------
    def _log_loss_fixed(self, y_true: np.ndarray, p: np.ndarray, *, n_classes: int, class_min: int) -> float:
        y = y_true.astype(int)
        y_idx = y - int(class_min)
        y_idx = np.clip(y_idx, 0, n_classes - 1)
        return float(log_loss(y_idx, p, labels=np.arange(n_classes)))

    def _brier_multiclass_fixed(self, y_true: np.ndarray, p: np.ndarray, *, n_classes: int, class_min: int) -> float:
        y = y_true.astype(int)
        y_idx = y - int(class_min)
        y_idx = np.clip(y_idx, 0, n_classes - 1)

        onehot = np.zeros_like(p, dtype=np.float64)
        onehot[np.arange(len(y_idx)), y_idx] = 1.0
        # multiclass Brier: mean over samples of sum_k (p_k - y_k)^2
        return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))

    # -----------------------------
    # Sampling utilities
    # -----------------------------
    def _apply_temperature(self, p: np.ndarray, temperature: float) -> np.ndarray:
        T = float(temperature)
        if T <= 0:
            raise ValueError("temperature must be > 0")
        if abs(T - 1.0) < 1e-12:
            return self._renorm(p)

        p = np.clip(p, 1e-12, None)
        logits = np.log(p) / T
        logits = logits - np.max(logits)
        out = np.exp(logits)
        return self._renorm(out)

    def _renorm(self, p: np.ndarray) -> np.ndarray:
        s = float(np.sum(p))
        if s <= 0:
            # fallback to uniform
            return np.ones_like(p, dtype=np.float64) / len(p)
        return (p / s).astype(np.float64)