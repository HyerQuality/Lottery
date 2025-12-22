import re
from typing import Optional, Sequence

import pandas as pd
from dateutil import parser


class Jackpots:
    """
    Scrape Powerball jackpot history from powerball.net and format it to match this project's
    jackpots.csv schema.

    Output DataFrame schema (matches jackpots.csv):
      - "Draw Date":  m/d/yyyy (no leading zeros)
      - "Jackpot":    string with commas and padded with spaces, e.g. " 564,100,000 "
      - "Winners":    int

    Notes
    -----
    - This class is intentionally conservative: it relies on pandas.read_html() and attempts
      to locate the table containing ["Draw Date", "Jackpot", "Winners"].
    - Website structure can change; errors are raised with contextual messages.
    """

    URL = "https://www.powerball.net/jackpots"
    _EXPECTED_COLS: Sequence[str] = ("Draw Date", "Jackpot", "Winners")

    def __init__(self, url: Optional[str] = None, since: str = "2015-01-01") -> None:
        self.url = url or self.URL
        # Preserve current behavior: interpret `since` via pandas and store a python date.
        self.since = pd.Timestamp(since).date()

    @staticmethod
    def clean_date(s: str):
        """
        Normalize a draw date string into a python `date`.

        Handles:
          - Ordinal suffixes: "1st", "2nd", "3rd", "4th" -> "1", "2", "3", "4"
          - Weekday prefixes: "Monday January 1, 2020" -> "January 1, 2020"
        """
        s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", str(s), flags=re.I)
        s = re.sub(
            r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+",
            "",
            s,
            flags=re.I,
        ).strip()
        return parser.parse(s, fuzzy=True).date()

    @staticmethod
    def clean_jackpot(s: str) -> str:
        """
        Normalize a jackpot string into a comma-formatted integer string, e.g. "564,100,000".

        Removes ▲▼ markers and all non-digits before parsing.
        Raises ValueError with context if no digits are found.
        """
        s = re.sub(r"[▲▼]", "", str(s))
        digits = re.sub(r"[^0-9]", "", s)
        if digits == "":
            raise ValueError(f"Unable to parse jackpot amount from value: {s!r}")
        return f"{int(digits):,}"

    @staticmethod
    def _normalize_columns(cols) -> list[str]:
        # Flatten potential MultiIndex headers and strip whitespace.
        if isinstance(cols, pd.MultiIndex):
            cols = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in cols.to_list()]
        return [str(c).strip() for c in cols]

    def _is_target_table(self, df: pd.DataFrame) -> bool:
        cols = self._normalize_columns(df.columns)
        return cols == list(self._EXPECTED_COLS)

    def run(self) -> pd.DataFrame:
        """
        Scrape, parse, filter, and format jackpot history.

        Returns
        -------
        pd.DataFrame
            Columns: ["Draw Date", "Jackpot", "Winners"], sorted ascending by draw date.
        """
        try:
            tables = pd.read_html(self.url)
        except Exception as e:
            raise RuntimeError(f"Failed to read HTML tables from {self.url!r}: {e}") from e

        rows: list[tuple] = []
        found_any = False

        for t in tables:
            # Normalize headers for comparison, but preserve original df for row access.
            if not self._is_target_table(t):
                continue

            found_any = True

            # Ensure we access by the expected column labels.
            # If pandas returned headers with whitespace, reassign normalized names.
            t = t.copy()
            t.columns = self._normalize_columns(t.columns)

            for _, r in t.iterrows():
                d = self.clean_date(r["Draw Date"])
                j = self.clean_jackpot(r["Jackpot"])

                # More robust than int(...) on possibly NaN/blank.
                w = pd.to_numeric(r["Winners"], errors="coerce")
                winners = int(w) if pd.notna(w) else 0

                rows.append((d, j, winners))

        if not found_any:
            raise RuntimeError(
                "Could not find jackpots table with columns "
                f"{list(self._EXPECTED_COLS)} at {self.url!r}. Website format may have changed."
            )

        df = pd.DataFrame(rows, columns=["date", "Jackpot", "Winners"])
        df = df[df["date"] >= self.since].copy()

        # Sort by true date for correctness; then emit the project's required string format.
        df = df.sort_values("date")

        # Match your project's jackpots.csv formatting exactly:
        df["Draw Date"] = df["date"].apply(lambda x: f"{x.month}/{x.day}/{x.year}")
        df["Jackpot"] = df["Jackpot"].apply(lambda x: f" {x} ")
        df = df[["Draw Date", "Jackpot", "Winners"]]

        return df

    def to_csv(self, out_path: str = "jackpots.csv") -> str:
        """Convenience helper: run() and write to a CSV with index=False."""
        df = self.run()
        df.to_csv(out_path, index=False)
        return out_path
