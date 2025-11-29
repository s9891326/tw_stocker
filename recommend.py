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
import twstock
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
    # 1. 讀取並標準化欄位名稱
    df = pd.read_csv(
        path_or_url,
        index_col="datetime",
        parse_dates=["datetime"],  # 讓 C-Parser 處理時間字串
    )
    df.rename(columns=lambda c: str(c).lower(), inplace=True)

    # 2. 檢查缺失欄位
    missing = sorted(REQUIRED_COLS - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns {missing} in: {path_or_url}")

    # 3. 數值轉換 (使用 REQUIRED_COLS 確保一致性)
    # 假設 REQUIRED_COLS 包含所有需要轉數值的欄位
    for c in REQUIRED_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. 清洗：濾除 Close 欄位為 NaN 的數據，並處理重複索引
    # 使用 .loc 避免 SettingWithCopyWarning
    df = df.loc[~df["close"].isna()]
    df = df.loc[~df.index.duplicated(keep="last")]

    # 5. 時區處理
    df.index = _tz_aware_daily_index(df.index)
    return df.sort_index()


import argparse

from strategy import institutional_buying


def _run_strategy(
    df: pd.DataFrame, parameters: Dict, strategy_name: str = "grid"
) -> Tuple[List[int], List[int], List[bool], List[bool], float, float]:
    """Run the selected strategy and return standardized outputs."""

    if strategy_name == "institutional":
        # Institutional Buying Strategy
        # Extract specific parameters if needed, or pass all
        states_buy, states_sell, states_entry, states_exit, total_gains, invest = (
            institutional_buying.trade(df, **parameters)
        )
    else:
        # Default to Grid Strategy
        states_buy, states_sell, states_entry, states_exit, total_gains, invest = (
            grid_trade(df, **parameters)
        )

    n = len(df)
    # Ensure all lists are of length n (padding if necessary, though strategies should handle this)
    # ... (existing padding logic) ...
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


def recommend_stock(
    path_or_url: str, parameters: Dict, strategy_name: str = "grid"
) -> Tuple[bool, bool, float, float]:
    """Run strategy on one instrument and derive recommendation.

    Returns: (should_buy, should_sell, today_close_price, total_gains)
    """
    df = _read_and_clean_csv(path_or_url)

    states_buy, states_sell, states_entry, states_exit, total_gains, invest = (
        _run_strategy(df, parameters, strategy_name)
    )

    today = len(df) - 1
    today_close_price = float(df["close"].iloc[-1])

    # Recommendation rule: signal happened in the last N bars (configurable)
    signal_window = int(parameters.get("signal_window", 5))
    should_buy = _recent_signal(today, states_buy, window=signal_window)
    should_sell = _recent_signal(today, states_sell, window=signal_window)

    return should_buy, should_sell, today_close_price, float(total_gains)


def _symbol_to_display(symbol: str) -> str:
    # Map numeric ticker (e.g., '2330') to its company name via twstock.codes
    try:
        meta = twstock.codes.get(symbol)
        if meta and getattr(meta, "name", None):
            return f"{meta.name}({symbol})"
    except Exception:
        pass
    return symbol


def generate_report(
    urls: Iterable[str | Path],
    parameters: Dict,
    limit: int = 10,
    output_path: str | Path = "stock_report.html",
    strategy_name: str = "grid",
) -> None:
    """Aggregate recommendations and render an HTML report using Jinja2.

    - Sort by Total_Gains desc and limit to top N.
    - Template: templates/stock_report_template.html
    """
    results: List[Dict] = []

    for url in urls:
        try:
            should_buy, should_sell, today_close_price, total_gains = recommend_stock(
                str(url), parameters, strategy_name
            )
            if should_sell or should_buy:
                stock_code = str(url).split("/")[-1].split(".")[0]
                if isinstance(url, Path):
                    stock_code = url.parts[-1].split(".")[0]
                stock_name = _symbol_to_display(stock_code)
                print(
                    f"{stock_name} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}, total_gains:{total_gains}"
                )
                results.append(
                    {
                        "Stock": stock_name,
                        "Should_Buy": bool(should_buy),
                        "Should_Sell": bool(should_sell),
                        "Recommended_Price": float(today_close_price),
                        "Total_Gains": float(total_gains),
                    }
                )
        except Exception:
            # Robust error handling but keep running others
            traceback.print_exc(file=sys.stderr)
            continue

    # Sort and limit
    sorted_results = sorted(
        results,
        key=lambda x: (x.get("Total_Gains") is None, x.get("Total_Gains", 0)),
        reverse=True,
    )[:limit]

    df = pd.DataFrame(sorted_results)
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("stock_report_template.html")
    html_output = template.render(
        stocks=df.to_dict(orient="records"), strategy=strategy_name
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_output)


def _iter_data_files(root: str | Path = "data") -> List[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted([p for p in root_path.glob("*.csv") if p.is_file()])


def parse_args():
    parser = argparse.ArgumentParser(description="TW Stocker Recommendation Engine")
    parser.add_argument(
        "--strategy",
        type=str,
        default="institutional",
        choices=["grid", "institutional"],
        help="Strategy to use: 'grid' or 'institutional'",
    )
    parser.add_argument(
        "--days", type=int, default=3, help="Continuous days for institutional strategy"
    )
    parser.add_argument(
        "--window", type=int, default=1, help="Signal window for institutional strategy"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    parameters: Dict = {
        "rsi_period": 14,
        "low_rsi": 30,
        "high_rsi": 70,
        "ema_period": 26,
        "continuous_days": args.days,
        "signal_window": args.window,
    }

    data_files = _iter_data_files("data")
    print(f"Running strategy: {args.strategy}...")

    # for p in data_files:
    #     try:
    #         should_buy, should_sell, today_close_price, total_gains = recommend_stock(
    #             str(p), parameters, args.strategy
    #         )
    #         if should_sell or should_buy:
    #             stock = p.stem
    #             print(
    #                 f"{stock} Should buy today: {should_buy}, Should sell today: {should_sell}, Recommended price: {today_close_price}, total_gains:{total_gains}"
    #             )
    #     except Exception:
    #         traceback.print_exc(file=sys.stderr)
    #         continue

    generate_report(
        data_files,
        parameters,
        limit=10,
        output_path="stock_report.html",
        strategy_name=args.strategy,
    )


if __name__ == "__main__":
    main()
