"""TW Stocker - Recommendation & Report Generator (daily, Asia/Taipei)

This module loads daily OHLCV CSVs from data/ (or given URLs/paths),
cleans and standardizes data, runs a trading strategy, aggregates
results, and renders an HTML report via Jinja2.

- Timezone: Asia/Taipei (tz-aware)
- Frequency: 1D (daily)
- CSV schema: at least [datetime/date, open, high, low, close, volume]
- Strategy interface: trade(df, **params) -> (states_buy, states_sell, states_entry, states_exit, total_gains, invest)
- Report template: templates/stock_report_template.html -> stock_report.html

Usage (script):
- Enumerates CSVs under data/ and generates stock_report.html

"""
import sys
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pytz
from jinja2 import Environment, FileSystemLoader

from strategy.grid import trade as grid_trade

TAIPEI_TZ = pytz.timezone("Asia/Taipei")
DAILY_FREQ = "1D"
REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def _tz_aware_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Ensure index is tz-aware (Asia/Taipei) and daily frequency.

    - If naive, localize to Asia/Taipei.
    - If tz-aware but not Asia/Taipei, convert to Asia/Taipei.
    - Normalize to date (keep date component) to represent daily bars.
    """
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")

    if idx.tz is None:
        idx = idx.tz_localize(TAIPEI_TZ)
    else:
        idx = idx.tz_convert(TAIPEI_TZ)

    idx = idx.normalize()
    return idx


def _read_and_clean_csv(path_or_url: str | Path) -> pd.DataFrame:
    """Read CSV, standardize columns to lowercase, coerce numerics, set tz-aware daily index.

    Accepts both local paths and URLs.
    """
    potential_index_cols = ["Datetime", "datetime", "date", "Date"]
    df: pd.DataFrame | None = None

    for idx_col in potential_index_cols:
        try:
            df = pd.read_csv(path_or_url, index_col=idx_col)
            break
        except Exception:
            continue

    if df is None:
        df = pd.read_csv(path_or_url)
        dt_candidate = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                dt_candidate = c
                break
        if dt_candidate is None:
            for c in potential_index_cols:
                if c in df.columns:
                    dt_candidate = c
                    break
        if dt_candidate is None:
            raise ValueError(f"No datetime column found in: {path_or_url}")
        df = df.set_index(dt_candidate)

    df.rename(columns=lambda c: str(c).lower(), inplace=True)

    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in: {path_or_url}")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[~df["close"].isna()].copy()

    df.index = _tz_aware_daily_index(df.index)

    df = df[~df.index.duplicated(keep="last")]

    df.sort_index(inplace=True)

    daily = df.resample('1D').last()
    daily = daily[~daily.index.duplicated(keep='last')]
    daily.sort_index(inplace=True)
    return daily


def _run_strategy(df: pd.DataFrame, parameters: Dict) -> Tuple[List[int], List[int], List[bool], List[bool], float, float]:
    """Run the selected strategy (grid by default) and return standardized outputs."""
    states_buy, states_sell, states_entry, states_exit, total_gains, invest = grid_trade(df, **parameters)

    n = len(df)
    if len(states_entry) != n:
        if len(states_entry) < n:
            states_entry = states_entry + [False] * (n - len(states_entry))
        else:
            states_entry = states_entry[:n]
    if len(states_exit) != n:
        if len(states_exit) < n:
            states_exit = states_exit + [False] * (n - len(states_exit))
        else:
            states_exit = states_exit[:n]

    return states_buy, states_sell, states_entry, states_exit, total_gains, invest


def _recent_signal(today_idx: int, indices: List[int], window: int = 5) -> bool:
    """Whether the last signal index occurred within a recent window from today."""
    if not indices:
        return False
    return abs(today_idx - indices[-1]) < window


def recommend_stock(path_or_url: str, parameters: Dict) -> Tuple[bool, bool, float, float]:
    """Run strategy on one instrument and derive recommendation.

    Returns: (should_buy, should_sell, today_close_price, total_gains)
    """
    df = _read_and_clean_csv(path_or_url)

    states_buy, states_sell, states_entry, states_exit, total_gains, invest = _run_strategy(df, parameters)

    today = len(df) - 1
    today_close_price = float(df["close"].iloc[-1])

    # Recommendation rule: signal happened in the last N bars (configurable)
    signal_window = int(parameters.get("signal_window", 5))
    should_buy = _recent_signal(today, states_buy, window=signal_window)
    should_sell = _recent_signal(today, states_sell, window=signal_window)

    return should_buy, should_sell, today_close_price, float(total_gains)


def generate_report(urls: Iterable[str | Path], parameters: Dict, limit: int = 10, output_path: str | Path = "stock_report.html") -> None:
    """Aggregate recommendations and render an HTML report using Jinja2.

    - Sort by Total_Gains desc and limit to top N.
    - Template: templates/stock_report_template.html
    """
    results: List[Dict] = []

    for url in urls:
        try:
            should_buy, should_sell, today_close_price, total_gains = recommend_stock(str(url), parameters)
            if should_sell or should_buy:
                stock_name = str(url).split("/")[-1].split(".")[0]
                results.append({
                    "Stock": stock_name,
                    "Should_Buy": bool(should_buy),
                    "Should_Sell": bool(should_sell),
                    "Recommended_Price": float(today_close_price),
                    "Total_Gains": float(total_gains),
                })
        except Exception:
            # Robust error handling but keep running others
            traceback.print_exc(file=sys.stderr)
            continue

    # Sort and limit
    sorted_results = sorted(results, key=lambda x: (x.get("Total_Gains") is None, x.get("Total_Gains", 0)), reverse=True)[:limit]

    df = pd.DataFrame(sorted_results)
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("stock_report_template.html")
    html_output = template.render(stocks=df.to_dict(orient="records"))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)


def _iter_data_files(root: str | Path = "data") -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted([p for p in root_path.glob("*.csv") if p.is_file()])


def main() -> None:
    # Default parameters for grid strategy; can be overridden
    parameters: Dict = {
        "rsi_period": 14,
        "low_rsi": 30,
        "high_rsi": 70,
        "ema_period": 26,
        # recommendation behavior
        # "signal_window": 5,
    }

    data_files = _iter_data_files("data")

    # Print picks to stdout
    # for p in data_files:
    #     try:
    #         should_buy, should_sell, today_close_price, total_gains = recommend_stock(str(p), parameters)
    #         if should_sell or should_buy:
    #             stock = p.stem
    #             print(f"{stock} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}, total_gains:{total_gains}")
    #     except Exception:
    #         traceback.print_exc(file=sys.stderr)
    #         continue

    # Generate report
    generate_report(data_files, parameters, limit=10, output_path="stock_report.html")


if __name__ == "__main__":
    main()
