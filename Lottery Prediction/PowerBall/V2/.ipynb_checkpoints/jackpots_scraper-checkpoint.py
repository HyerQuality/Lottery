import re
import pandas as pd
from dateutil import parser

class Jackpots:
    URL = "https://www.powerball.net/jackpots"

    def __init__(self, url: str | None = None, since: str = "2015-01-01"):
        self.url = url or self.URL
        self.since = pd.Timestamp(since).date()

    @staticmethod
    def clean_date(s: str):
        s = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", str(s), flags=re.I)
        s = re.sub(
            r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+",
            "",
            s,
            flags=re.I,
        ).strip()
        return parser.parse(s, fuzzy=True).date()

    @staticmethod
    def clean_jackpot(s: str):
        s = re.sub(r"[▲▼]", "", str(s))
        digits = re.sub(r"[^0-9]", "", s)
        return f"{int(digits):,}"

    def run(self) -> pd.DataFrame:
        tables = pd.read_html(self.url)

        rows = []
        for t in tables:
            if list(t.columns) == ["Draw Date", "Jackpot", "Winners"]:
                for _, r in t.iterrows():
                    d = self.clean_date(r["Draw Date"])
                    j = self.clean_jackpot(r["Jackpot"])
                    w = int(r["Winners"])
                    rows.append((d, j, w))

        df = pd.DataFrame(rows, columns=["date", "Jackpot", "Winners"])
        df = df[df["date"] >= self.since].copy()

        # Match your project's jackpots.csv formatting
        df["Draw Date"] = df["date"].apply(lambda x: f"{x.month}/{x.day}/{x.year}")
        df["Jackpot"] = df["Jackpot"].apply(lambda x: f" {x} ")
        df = df[["Draw Date", "Jackpot", "Winners"]].sort_values(
            "Draw Date", key=pd.to_datetime
        )

        return df

    def to_csv(self, out_path: str = "jackpots.csv") -> str:
        df = self.run()
        df.to_csv(out_path, index=False)
        return out_path
