import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import twstock

TAIPEI_TZ = pytz.timezone("Asia/Taipei")
DATA_DIR = "./data/"
CSV_COLUMNS = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
]  # base price schema
# Optional institutional columns (if available in data/institutional/<symbol>.csv)
INSTITUTIONAL_COLS = ["foreign_net_buy", "invest_trust_net_buy", "dealer_net_buy"]
DEFAULT_LOOKBACK_MONTHS = 6  # initial backfill window when no CSV exists
DEFAULT_WORKERS = 4
DEFAULT_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5

CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Cross-process TWSE rate limit configuration
# Increased to 3.0s to avoid IP blocking on heavy T86 requests
TWSE_MIN_INTERVAL_SEC = float(os.getenv("TWSE_MIN_INTERVAL_SEC", "3.0"))
_TWSE_LOCK_PATH = os.path.join(CACHE_DIR, "twse_request.lock")
_TWSE_TS_PATH = os.path.join(CACHE_DIR, "twse_last_ts")

# User-Agent rotation list
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
]


def ensure_data_dir(path: str = DATA_DIR) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clean_csv(csv_path: str) -> None:
    """Clean the CSV file by removing rows with incorrect number of columns."""
    if not os.path.exists(csv_path):
        return
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            return
        expected_cols = len(lines[0].rstrip("\n").split(","))
        with open(csv_path, "w", encoding="utf-8") as f:
            for line in lines:
                if len(line.rstrip("\n").split(",")) == expected_cols:
                    f.write(line)
    except Exception as e:
        print(f"Warning: failed to clean CSV {csv_path}: {e}")


def month_range(start: date, end: date) -> List[Tuple[int, int]]:
    """Generate inclusive (year, month) tuples from start to end (by calendar month)."""
    if start > end:
        return []
    y, m = start.year, start.month
    result = []
    while (y < end.year) or (y == end.year and m <= end.month):
        result.append((y, m))
        # increment month
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return result


def _to_tz_aware(dt: datetime | date) -> datetime:
    if isinstance(dt, date) and not isinstance(dt, datetime):
        # interpret as market day at 00:00 Asia/Taipei
        naive = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
    else:
        naive = dt  # type: ignore[assignment]
    if naive.tzinfo is None:
        return TAIPEI_TZ.localize(naive)
    return naive.astimezone(TAIPEI_TZ)


def _retry_fetch_month(
    stock_num: str, year: int, month: int, retries: int = DEFAULT_RETRIES
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            s = twstock.Stock(stock_num)
            records = s.fetch(year, month)
            rows = []
            for r in records:
                dt_tz = _to_tz_aware(r.date)
                rows.append(
                    {
                        "datetime": dt_tz,
                        "open": r.open,
                        "high": r.high,
                        "low": r.low,
                        "close": r.close,
                        "volume": r.capacity,
                    }
                )
            df = pd.DataFrame(rows, columns=CSV_COLUMNS)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            last_exc = e
            # exponential backoff
            time.sleep((RETRY_BACKOFF_SEC ** (i + 1)) + (0.1 * i))
    # final attempt failed
    raise RuntimeError(
        f"Failed to fetch {stock_num} {year}-{month:02d} after {retries} retries: {last_exc}"
    )


def _fmt_yyyymmdd(dt: date | datetime) -> str:
    if isinstance(dt, datetime):
        d = dt.date()
    else:
        d = dt
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


def _parse_int(val: str | int | float | None) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return int(val)
        except Exception:
            return None
    s = str(val).strip().replace(",", "")
    if s in ("", "--"):
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _acquire_file_lock(lock_path: str, timeout: float = 30.0) -> Optional[int]:
    """Acquire an inter-process file lock using O_CREAT|O_EXCL.

    Returns a file descriptor on success, or None on timeout/failure.
    This is a simple best-effort lock (works across processes) without extra deps.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            # Write PID and timestamp for debugging
            try:
                os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode("utf-8"))
            except Exception:
                pass
            return fd
        except FileExistsError:
            time.sleep(random.uniform(1, 3))
        except Exception:
            break
    return None


def _release_file_lock(fd: Optional[int], lock_path: str) -> None:
    try:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def _twse_rate_limit_sleep(min_interval: float = TWSE_MIN_INTERVAL_SEC) -> None:
    """Best-effort cross-process rate limiter using a lock + timestamp file.

    Ensures at least `min_interval` seconds between any two TWSE requests
    among all cooperating processes using the same cache dir.
    """
    fd = _acquire_file_lock(_TWSE_LOCK_PATH, timeout=10.0)
    # Even if we can't acquire the lock, still try to be polite with a small sleep
    if fd is None:
        time.sleep(min(0.2, min_interval))
        return
    try:
        now = time.time()
        last = None
        try:
            if os.path.exists(_TWSE_TS_PATH):
                with open(_TWSE_TS_PATH, "r", encoding="utf-8") as f:
                    s = f.read().strip()
                    if s:
                        last = float(s)
        except Exception:
            last = None
        if last is not None:
            gap = now - last
            if gap < min_interval:
                time.sleep(min_interval - gap + random.uniform(0, 0.05))
        # Update timestamp to just-before making request
        try:
            with open(_TWSE_TS_PATH, "w", encoding="utf-8") as f:
                f.write(str(time.time()))
        except Exception:
            pass
    finally:
        _release_file_lock(fd, _TWSE_LOCK_PATH)


def _cache_key(path: str, params: dict) -> str:
    qs = urllib.parse.urlencode(sorted(params.items()))
    safe = path.strip("/").replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe}_{qs}.json")


def _twse_get_json(
    path: str, params: dict, *, use_cache: bool = True, max_retries: int = 3
) -> dict:
    base = "https://www.twse.com.tw"
    qs = urllib.parse.urlencode(params)
    url = f"{base}{path}?{qs}"

    cache_file = _cache_key(path, params)

    def double_check_cache(use_cache: bool = True):
        # Double-check cache after acquiring lock
        if use_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    print(f"Cache hit after wait: {cache_file}")
                    return json.load(f)
            except Exception:
                pass
        return None

    # First check without lock for speed
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        fd = None
        try:
            # Acquire lock to serialize requests and prevent redundant fetching
            # We use the same lock file for all TWSE requests to respect rate limits AND prevent race conditions
            fd = _acquire_file_lock(_TWSE_LOCK_PATH, timeout=30.0)

            result = double_check_cache(use_cache)
            if result:
                return result

            # Check rate limit timestamp
            now = time.time()
            last = None
            try:
                if os.path.exists(_TWSE_TS_PATH):
                    with open(_TWSE_TS_PATH, "r", encoding="utf-8") as f:
                        s = f.read().strip()
                        if s:
                            last = float(s)
            except Exception:
                pass

            if last is not None:
                gap = now - last
                if gap < TWSE_MIN_INTERVAL_SEC:
                    time.sleep(TWSE_MIN_INTERVAL_SEC - gap + random.uniform(0, 0.05))

            result = double_check_cache(use_cache)
            if result:
                return result

            # Fetch
            ua = random.choice(USER_AGENTS)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": ua,
                    "Referer": "https://www.twse.com.tw/zh/page/trading/fund/T86.html",
                    "Accept": "application/json, text/javascript, */*; q=0.01",
                    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                    "Connection": "keep-alive",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"TWSE request URL: {url} (status={resp.status})")
                data = resp.read()

            # Update timestamp
            try:
                with open(_TWSE_TS_PATH, "w", encoding="utf-8") as f:
                    f.write(str(time.time()))
            except Exception:
                pass

            js = json.loads(data)
            if use_cache:
                try:
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(js, f, ensure_ascii=False)
                except Exception:
                    pass

            return js

        except Exception as e:
            if attempt == max_retries:
                print(f"TWSE request failed after {max_retries} attempts: {e}")
                break
            # jittered exponential backoff
            sleep_s = backoff + random.uniform(0, 0.5)
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 8)
        finally:
            _release_file_lock(fd, _TWSE_LOCK_PATH)

    return {}


def _fetch_t86_by_date(yyyymmdd: str) -> pd.DataFrame:
    # 三大法人買賣超個股 統計表 (T86)
    # Example: /rwd/zh/fund/T86?response=json&date=20240102&selectType=ALLBUT0999
    payload = {
        "response": "json",
        "date": yyyymmdd,
        "selectType": "ALLBUT0999",
    }
    js = _twse_get_json("/rwd/zh/fund/T86", payload)
    fields = js.get("fields") or []
    data = js.get("data") or []

    if not fields or not data:
        return pd.DataFrame(columns=["code", "foreign", "invest", "dealer"])

    df = pd.DataFrame(data, columns=fields)
    out = pd.DataFrame(
        {
            "code": df["證券代號"].astype(str).str.strip(),
            "foreign": df["外陸資買賣超股數(不含外資自營商)"].apply(_parse_int),
            "invest": df["投信買賣超股數"].apply(_parse_int),
            "dealer": df["自營商買賣超股數"].apply(_parse_int),
        }
    )
    out["date"] = pd.to_datetime(yyyymmdd, format="%Y%m%d").tz_localize(TAIPEI_TZ)
    out.rename(
        columns={
            "foreign": "foreign_net_buy",
            "invest": "invest_trust_net_buy",
            "dealer": "dealer_net_buy",
        },
        inplace=True,
    )
    return out[
        ["date", "code", "foreign_net_buy", "invest_trust_net_buy", "dealer_net_buy"]
    ]


def _fetch_institutional_for_dates(
    symbol: str, dates: Iterable[date | datetime]
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    # Sort dates to ensure sequential processing (helps with cache locality if any)
    sorted_dates = sorted(dates)

    for dt in sorted_dates:
        yyyymmdd = _fmt_yyyymmdd(dt)
        try:
            t86 = _fetch_t86_by_date(yyyymmdd)
            if not t86.empty:
                sub = t86.loc[t86["code"] == str(symbol)].copy()
                if not sub.empty:
                    sub.rename(columns={"date": "datetime"}, inplace=True)
                    rows.append(
                        sub[
                            [
                                "datetime",
                                "foreign_net_buy",
                                "invest_trust_net_buy",
                                "dealer_net_buy",
                            ]
                        ]
                    )
        except Exception as e:
            # keep going; network or schema errors shouldn't stop whole update
            print(f"Warning: failed to fetch T86 for {symbol} on {yyyymmdd}: {e}")
            continue
    if not rows:
        return pd.DataFrame(columns=["datetime"] + INSTITUTIONAL_COLS)
    df = pd.concat(rows, ignore_index=True)
    df = (
        df.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    return df


def load_existing(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    clean_csv(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Parser error when reading CSV {csv_path}: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)

    df.columns = [c.strip().lower() for c in df.columns]

    # Remove corrupted columns from previous bad merges
    cols_to_keep = [
        c for c in df.columns if not c.endswith("_x") and not c.endswith("_y")
    ]
    df = df[cols_to_keep]

    if "datetime" not in df.columns:
        if "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})
        else:
            return pd.DataFrame(columns=CSV_COLUMNS)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["datetime"] = df["datetime"].apply(
        lambda x: _to_tz_aware(x.to_pydatetime()) if pd.notna(x) else x
    )

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = pd.Series(dtype="float64")
    # Keep all existing columns (including previously added institutional fields)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def _calc_lookback_start(today: date, lookback_months: int) -> date:
    # compute the first day of N months before today
    year = today.year
    month = today.month - lookback_months
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


def get_daily_data_since_last_record(
    stock_num: str,
    base_path: str = DATA_DIR,
    *,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
) -> pd.DataFrame:
    ensure_data_dir(base_path)
    csv_path = os.path.join(base_path, f"{stock_num}.csv")

    existing = load_existing(csv_path)

    today = datetime.now(TAIPEI_TZ).date()

    if existing.empty:
        start = _calc_lookback_start(today, lookback_months)
    else:
        last_dt: datetime = existing["datetime"].iloc[-1]
        last_day = last_dt.date()
        next_day = last_day + timedelta(days=1)
        start = date(next_day.year, next_day.month, 1)

    if (not existing.empty) and start > today:
        pass

    frames: List[pd.DataFrame] = []
    # Only fetch new price data if needed
    if start <= today:
        for y, m in month_range(start, today):
            try:
                dfm = _retry_fetch_month(stock_num, y, m)
                frames.append(dfm)
            except Exception as e:
                print(f"Warning: failed to fetch {stock_num} {y}-{m:02d}: {e}")

    new_df = pd.DataFrame(columns=CSV_COLUMNS)
    if frames:
        new_df = pd.concat(frames, ignore_index=True)

    # only keep strictly new rows compared to existing
    if not existing.empty and not new_df.empty:
        last_dt = existing["datetime"].iloc[-1]
        new_df = new_df[new_df["datetime"] > last_dt]

    if not new_df.empty:
        new_df = new_df.dropna(subset=["close"])
        new_df = (
            new_df.drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )

    # Combine existing price data with new price data first
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )

    # Decide which dates to fetch institutional data for
    # We want to fetch for new data AND potentially backfill recent missing data
    # For simplicity, let's fetch for the new_df dates + last 5 days of existing
    dates_for_fetch = pd.Series(dtype="datetime64[ns, Asia/Taipei]")

    if not new_df.empty:
        dates_for_fetch = pd.concat([dates_for_fetch, new_df["datetime"]])

    if not existing.empty:
        # Check last few days of existing to see if they have institutional data
        # If not, add them to fetch list
        tail = existing.tail(10)
        missing_inst = (
            tail[tail[INSTITUTIONAL_COLS[0]].isna()]
            if INSTITUTIONAL_COLS[0] in tail.columns
            else tail
        )
        if not missing_inst.empty:
            dates_for_fetch = pd.concat([dates_for_fetch, missing_inst["datetime"]])
        else:
            # Always re-check last 3 days just in case data wasn't available then
            dates_for_fetch = pd.concat([dates_for_fetch, existing["datetime"].tail(3)])

    dates_for_fetch = dates_for_fetch.drop_duplicates().sort_values()

    if not dates_for_fetch.empty:
        inst_df = _fetch_institutional_for_dates(stock_num, dates_for_fetch)

        if not inst_df.empty:
            # Use update logic instead of merge to avoid duplicates
            # Ensure columns exist
            for col in INSTITUTIONAL_COLS:
                if col not in combined.columns:
                    combined[col] = np.nan

            # Set index for update
            combined.set_index("datetime", inplace=True)
            inst_df.set_index("datetime", inplace=True)

            # Update combined with inst_df values
            combined.update(inst_df)

            # Reset index
            combined.reset_index(inplace=True)

    # Ensure columns exist if no fetch happened
    for c in INSTITUTIONAL_COLS:
        if c not in combined.columns:
            combined[c] = np.nan

    if not combined.empty:
        out_df = combined.copy()
        out_df["datetime"] = out_df["datetime"].apply(lambda x: x.isoformat())
        out_df.to_csv(csv_path, index=False, encoding="utf-8")

    # return only the newly added rows for progress reporting
    return new_df


def should_include_code(meta) -> bool:
    # Include 上市 股票 or ETF
    return (meta.market == "上市") and (meta.type in ("股票", "ETF"))


def _update_one(
    symbol: str,
    *,
    base_path: str = DATA_DIR,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
) -> tuple[str, int, Optional[str]]:
    try:
        updated = get_daily_data_since_last_record(
            symbol, base_path=base_path, lookback_months=lookback_months
        )
        return symbol, int(updated.shape[0]), None
    except Exception as e:
        return symbol, 0, str(e)


def _update_one_args(args: tuple[str, str, int]) -> tuple[str, int, Optional[str]]:
    symbol, base_path, lookback_months = args
    return _update_one(symbol, base_path=base_path, lookback_months=lookback_months)


def update_all(
    *, workers: int = DEFAULT_WORKERS, lookback_months: int = DEFAULT_LOOKBACK_MONTHS
) -> None:
    codes = twstock.codes
    symbols: List[str] = [k for k, v in codes.items() if should_include_code(v)]
    if not symbols:
        print("No symbols to update (filter may be too strict).")
        return

    print(
        f"Updating {len(symbols)} symbols with {workers} workers, lookback={lookback_months} months..."
    )

    args_list: List[tuple[str, str, int]] = [
        (s, DATA_DIR, lookback_months) for s in symbols
    ]

    # Run in parallel; each process writes to its own CSV to avoid write contention
    completed = 0
    updated_total = 0
    errors = []

    def _log_progress(sym: str, rows: int, err: Optional[str]):
        nonlocal completed, updated_total, errors
        completed += 1
        updated_total += rows
        if err:
            errors.append((sym, err))
            print(f"[{completed}/{len(symbols)}] {sym}: ERROR - {err}")
        else:
            if rows == 0:
                print(f"[{completed}/{len(symbols)}] {sym}: no new data")
            else:
                print(f"[{completed}/{len(symbols)}] {sym}: updated {rows} rows")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_sym = {
            executor.submit(_update_one_args, args): args[0] for args in args_list
        }
        for future in as_completed(future_to_sym):
            sym, rows, err = future.result()
            _log_progress(sym, rows, err)

    print(
        f"Done. Symbols: {len(symbols)}, total new rows: {updated_total}, errors: {len(errors)}"
    )
    if errors:
        print("Some symbols failed:")
        for sym, err in errors[:20]:  # cap output
            print(f" - {sym}: {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Update TW stock daily data (parallel, multithreading)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers (threads)",
    )
    p.add_argument(
        "--lookback-months",
        type=int,
        default=DEFAULT_LOOKBACK_MONTHS,
        help="Initial backfill months when CSV missing",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"workers={args.workers}, lookback_months={args.lookback_months}")
    workers = max(1, args.workers)
    lookback = max(1, args.lookback_months)
    update_all(workers=workers, lookback_months=lookback)
