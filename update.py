from datetime import datetime, date, timedelta
import os
from typing import List, Tuple

import pandas as pd
import pytz
import twstock


TAIPEI_TZ = pytz.timezone('Asia/Taipei')
DATA_DIR = './data/'
CSV_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume']

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


def fetch_twstock_month(stock_num: str, year: int, month: int) -> pd.DataFrame:
    """Fetch one month's daily OHLCV using twstock.

    Returns DataFrame with columns CSV_COLUMNS and tz-aware datetime index (not set as index yet).
    """
    s = twstock.Stock(stock_num)
    records = s.fetch(year, month)
    rows = []
    for r in records:
        # r has fields: date, capacity, turnover, open, high, low, close, change, transaction
        dt_tz = _to_tz_aware(r.date)
        row = {
            'datetime': dt_tz,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.capacity,
        }
        rows.append(row)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    # Normalize dtypes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
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


def get_daily_data_since_last_record(stock_num: str, base_path: str = DATA_DIR) -> pd.DataFrame:
    ensure_data_dir(base_path)
    csv_path = os.path.join(base_path, f'{stock_num}.csv')

    existing = load_existing(csv_path)

    if existing.empty:
        today = datetime.now(TAIPEI_TZ).date()
        start = date(today.year - 1, today.month, 1)
    else:
        last_dt: datetime = existing['datetime'].iloc[-1]
        last_day = last_dt.date()
        start = (last_day + timedelta(days=1)).replace(day=1)

    today = datetime.now(TAIPEI_TZ).date()

    if existing.shape[0] > 0 and start > today:
        return pd.DataFrame(columns=CSV_COLUMNS)

    frames: List[pd.DataFrame] = []
    for y, m in month_range(start, today):
        try:
            dfm = fetch_twstock_month(stock_num, y, m)
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
    # Include上市 股票 or ETF
    return (meta.market == '上市') and (meta.type in ('股票', 'ETF'))


def update_all() -> None:
    codes = twstock.codes
    for k, v in codes.items():
        try:
            if should_include_code(v):
                updated = get_daily_data_since_last_record(k)
                if updated.empty:
                    print(f"No new data for {k}")
                else:
                    print(f"Updated {k}: {len(updated)} rows")
        except Exception as e:
            print(f"Error updating {k}: {e}")


if __name__ == '__main__':
    update_all()
