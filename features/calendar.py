"""Calendar / event-proximity features.

- Days to / since next earnings
- Days to OPEX (3rd Friday of month)
- Days to FOMC (approximated as ~6-week cadence; replace with real calendar in prod)
- Quarter-end / month-end proximity
"""
from __future__ import annotations

import calendar as cal
from datetime import date, datetime, timedelta

import pandas as pd


# FOMC meeting dates 2022-2026 (approximate; update annually in production)
FOMC_DATES_KNOWN = [
    '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15', '2022-07-27',
    '2022-09-21', '2022-11-02', '2022-12-14',
    '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14', '2023-07-26',
    '2023-09-20', '2023-11-01', '2023-12-13',
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31',
    '2024-09-18', '2024-11-07', '2024-12-18',
    '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30',
    '2025-09-17', '2025-10-29', '2025-12-10',
    '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29',
    '2026-09-16', '2026-10-28', '2026-12-09',
]
FOMC_DATES = pd.DatetimeIndex(FOMC_DATES_KNOWN)


def _third_friday(year: int, month: int) -> date:
    """3rd Friday of a month — standard monthly options expiry."""
    c = cal.Calendar()
    fridays = [d for d in c.itermonthdates(year, month) if d.weekday() == 4 and d.month == month]
    return fridays[2]


def compute_calendar_features(
    index: pd.DatetimeIndex,
    next_earnings_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calendar features.

    Args:
        index: DatetimeIndex for the daily series
        next_earnings_date: best-known next earnings date (or None)

    Returns:
        DataFrame indexed by `index` with calendar features prefixed 'cal_'.
    """
    out = pd.DataFrame(index=index)

    # Days to / from earnings
    if next_earnings_date is not None:
        out['cal_days_to_earnings'] = (next_earnings_date - index).days
        # Clip to reasonable range
        out['cal_days_to_earnings'] = out['cal_days_to_earnings'].clip(-180, 180)
    else:
        out['cal_days_to_earnings'] = pd.NA
    out['cal_earnings_within_30'] = (
        (out['cal_days_to_earnings'] >= 0) & (out['cal_days_to_earnings'] <= 30)
    ).astype(int) if next_earnings_date is not None else 0
    out['cal_earnings_within_7'] = (
        (out['cal_days_to_earnings'] >= 0) & (out['cal_days_to_earnings'] <= 7)
    ).astype(int) if next_earnings_date is not None else 0

    # Days to next OPEX (3rd Friday of current or next month)
    opex_days = []
    for d in index:
        # Try this month
        try:
            this_opex = pd.Timestamp(_third_friday(d.year, d.month))
        except Exception:
            this_opex = None
        if this_opex is not None and d <= this_opex:
            opex_days.append((this_opex - d).days)
            continue
        # Next month
        next_month = d + pd.DateOffset(months=1)
        try:
            next_opex = pd.Timestamp(_third_friday(next_month.year, next_month.month))
            opex_days.append((next_opex - d).days)
        except Exception:
            opex_days.append(20)
    out['cal_days_to_opex'] = opex_days

    # Days to nearest FOMC (forward-looking)
    fomc_future = FOMC_DATES[FOMC_DATES >= index.min()]
    days_to_fomc = []
    for d in index:
        future = fomc_future[fomc_future >= d]
        if len(future) == 0:
            days_to_fomc.append(60)
        else:
            days_to_fomc.append((future[0] - d).days)
    out['cal_days_to_fomc'] = days_to_fomc
    out['cal_fomc_within_5'] = (out['cal_days_to_fomc'] <= 5).astype(int)

    # Day-of-week (weekday-effect feature)
    out['cal_dow'] = index.dayofweek

    # Month-end / quarter-end proximity
    out['cal_month_end_pos'] = index.day / index.days_in_month  # 0-1
    out['cal_is_quarter_end_week'] = (
        (index.month % 3 == 0) & (index.day >= 23)
    ).astype(int)

    return out
