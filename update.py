from datetime import datetime, date, timedelta
import os
import time
import argparse
from typing import List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import pandas as pd
import pytz
import twstock


TAIPEI_TZ = pytz.timezone('Asia/Taipei')
DATA_DIR = './data/'
CSV_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']
DEFAULT_LOOKBACK_MONTHS = 6  # initial backfill window when no CSV exists
DEFAULT_WORKERS = max(1, min(8, cpu_count() - 1))  # be conservative by default
DEFAULT_RETRIES = 3
RETRY_BACKOFF_SEC = 1.5


def ensure_data_dir(path: str = DATA_DIR) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def clean_csv(csv_path: str) -> None:
    """Clean the CSV file by removing rows with incorrect number of columns."""
    if not os.path.exists(csv_path):
        return
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return
        expected_cols = len(lines[0].rstrip('\n').split(','))
        with open(csv_path, 'w', encoding='utf-8') as f:
            for line in lines:
                if len(line.rstrip('\n').split(',')) == expected_cols:
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


def _retry_fetch_month(stock_num: str, year: int, month: int, retries: int = DEFAULT_RETRIES) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for i in range(retries):
        try:
            s = twstock.Stock(stock_num)
            records = s.fetch(year, month)
            rows = []
            for r in records:
                dt_tz = _to_tz_aware(r.date)
                rows.append({
                    'datetime': dt_tz,
                    'open': r.open,
                    'high': r.high,
                    'low': r.low,
                    'close': r.close,
                    'volume': r.capacity,
                })
            df = pd.DataFrame(rows, columns=CSV_COLUMNS)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            last_exc = e
            # exponential backoff
            time.sleep((RETRY_BACKOFF_SEC ** (i + 1)) + (0.1 * i))
    # final attempt failed
    raise RuntimeError(f"Failed to fetch {stock_num} {year}-{month:02d} after {retries} retries: {last_exc}")


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
    if 'datetime' not in df.columns:
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        elif 'time' in df.columns:
            df = df.rename(columns={'time': 'datetime'})
        else:
            return pd.DataFrame(columns=CSV_COLUMNS)

    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
    df['datetime'] = df['datetime'].apply(lambda x: _to_tz_aware(x.to_pydatetime()) if pd.notna(x) else x)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = pd.Series(dtype='float64')
    df = df[CSV_COLUMNS]
    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    return df


def _calc_lookback_start(today: date, lookback_months: int) -> date:
    # compute the first day of N months before today
    year = today.year
    month = today.month - lookback_months
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, 1)


def get_daily_data_since_last_record(stock_num: str, base_path: str = DATA_DIR, *, lookback_months: int = DEFAULT_LOOKBACK_MONTHS) -> pd.DataFrame:
    ensure_data_dir(base_path)
    csv_path = os.path.join(base_path, f'{stock_num}.csv')

    existing = load_existing(csv_path)

    today = datetime.now(TAIPEI_TZ).date()

    if existing.empty:
        start = _calc_lookback_start(today, lookback_months)
    else:
        last_dt: datetime = existing['datetime'].iloc[-1]
        last_day = last_dt.date()
        # next calendar month from the last recorded day
        next_day = last_day + timedelta(days=1)
        start = date(next_day.year, next_day.month, 1)

    if (not existing.empty) and start > today:
        return pd.DataFrame(columns=CSV_COLUMNS)

    frames: List[pd.DataFrame] = []
    for y, m in month_range(start, today):
        try:
            dfm = _retry_fetch_month(stock_num, y, m)
            frames.append(dfm)
        except Exception as e:
            print(f"Warning: failed to fetch {stock_num} {y}-{m:02d}: {e}")

    if not frames:
        return pd.DataFrame(columns=CSV_COLUMNS)

    new_df = pd.concat(frames, ignore_index=True)

    if not existing.empty:
        last_dt = existing['datetime'].iloc[-1]
        new_df = new_df[new_df['datetime'] > last_dt]

    new_df = new_df.dropna(subset=['close'])
    new_df = (
        new_df
        .drop_duplicates(subset=['datetime'])
        .sort_values('datetime')
        .reset_index(drop=True)
    )

    if not new_df.empty:
        write_df = new_df.copy()
        write_df['datetime'] = write_df['datetime'].apply(lambda x: x.isoformat())
        if os.path.exists(csv_path) and existing.shape[0] > 0:
            write_df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')
        else:
            write_df.to_csv(csv_path, index=False, encoding='utf-8')

    return new_df


def should_include_code(meta) -> bool:
    # Include 上市 股票 or ETF
    return (meta.market == '上市') and (meta.type in ('股票', 'ETF'))


def _update_one(symbol: str, *, base_path: str = DATA_DIR, lookback_months: int = DEFAULT_LOOKBACK_MONTHS) -> tuple[str, int, Optional[str]]:
    try:
        updated = get_daily_data_since_last_record(symbol, base_path=base_path, lookback_months=lookback_months)
        return symbol, int(updated.shape[0]), None
    except Exception as e:
        return symbol, 0, str(e)


def _update_one_args(args: tuple[str, str, int]) -> tuple[str, int, Optional[str]]:
    symbol, base_path, lookback_months = args
    return _update_one(symbol, base_path=base_path, lookback_months=lookback_months)


def update_all(*, workers: int = DEFAULT_WORKERS, lookback_months: int = DEFAULT_LOOKBACK_MONTHS) -> None:
    codes = twstock.codes
    symbols: List[str] = [k for k, v in codes.items() if should_include_code(v)]
    if not symbols:
        print('No symbols to update (filter may be too strict).')
        return

    print(f"Updating {len(symbols)} symbols with {workers} workers, lookback={lookback_months} months...")

    args_list: List[tuple[str, str, int]] = [(s, DATA_DIR, lookback_months) for s in symbols]

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

    with Pool(processes=workers) as pool:
        for sym, rows, err in pool.imap_unordered(_update_one_args, args_list):
            _log_progress(sym, rows, err)

    print(f"Done. Symbols: {len(symbols)}, total new rows: {updated_total}, errors: {len(errors)}")
    if errors:
        print("Some symbols failed:")
        for sym, err in errors[:20]:  # cap output
            print(f" - {sym}: {err}")
        if len(errors) > 20:
            print(f" ... and {len(errors) - 20} more")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Update TW stock daily data (parallel)')
    p.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='Number of parallel workers (processes)')
    p.add_argument('--lookback-months', type=int, default=DEFAULT_LOOKBACK_MONTHS, help='Initial backfill months when CSV missing')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"workers={args.workers}, lookback_months={args.lookback_months}")
    workers = max(1, args.workers)
    lookback = max(1, args.lookback_months)
    update_all(workers=workers, lookback_months=lookback)
