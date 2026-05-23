"""yfinance data fetcher with on-disk caching.

Caches OHLCV history as parquet per ticker and fundamentals as JSON.
Cache is keyed by (ticker, end_date) so different runs on the same day reuse data.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch OHLCV + fundamentals from yfinance with disk caching."""

    def __init__(self, cache_dir: str | Path = './.cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._ohlcv_cache = self.cache_dir / 'ohlcv'
        self._info_cache = self.cache_dir / 'info'
        self._ohlcv_cache.mkdir(exist_ok=True)
        self._info_cache.mkdir(exist_ok=True)

        # Lazy import yfinance
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise RuntimeError(
                "yfinance not installed. Run: pip install yfinance"
            )

    # ----------------------------------------------------------------- OHLCV

    def get_ohlcv(
        self,
        ticker: str,
        start: str | datetime,
        end: str | datetime | None = None,
        max_cache_age_hours: int = 24,
    ) -> pd.DataFrame:
        """Fetch OHLCV data for one ticker with caching.

        Args:
            ticker: stock symbol (e.g., 'NVDA')
            start: start date (string YYYY-MM-DD or datetime)
            end: end date (default: today)
            max_cache_age_hours: refresh cache if older than this

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume, AdjClose]
            indexed by date. Empty DataFrame on fetch failure.
        """
        end = end or datetime.now().strftime('%Y-%m-%d')
        if isinstance(start, datetime):
            start = start.strftime('%Y-%m-%d')
        if isinstance(end, datetime):
            end = end.strftime('%Y-%m-%d')

        cache_file = self._ohlcv_cache / f'{ticker}.parquet'
        cache_meta = self._ohlcv_cache / f'{ticker}.meta.json'

        # Check cache freshness
        if cache_file.exists() and cache_meta.exists():
            with open(cache_meta) as f:
                meta = json.load(f)
            cached_age = (datetime.now() - datetime.fromisoformat(meta['fetched_at'])).total_seconds() / 3600
            cached_end = meta.get('end')
            if cached_age < max_cache_age_hours and cached_end == end:
                try:
                    df = pd.read_parquet(cache_file)
                    df.index = pd.to_datetime(df.index)
                    return df.loc[start:end]
                except Exception as e:
                    logger.warning(f"Cache read failed for {ticker}: {e}")

        # Fetch fresh
        try:
            ticker_obj = self._yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start, end=end,
                auto_adjust=False, actions=False,
                interval='1d',
            )
            if df.empty:
                logger.warning(f"yfinance returned empty for {ticker}")
                return pd.DataFrame()

            # Normalize columns
            df = df.rename(columns={'Adj Close': 'AdjClose'})
            keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume'] if c in df.columns]
            df = df[keep_cols].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)

            # Cache
            df.to_parquet(cache_file)
            with open(cache_meta, 'w') as f:
                json.dump({
                    'ticker': ticker,
                    'fetched_at': datetime.now().isoformat(),
                    'start': start,
                    'end': end,
                    'rows': len(df),
                }, f)
            return df
        except Exception as e:
            logger.error(f"Fetch failed for {ticker}: {e}")
            return pd.DataFrame()

    # ---------------------------------------------------------------- BATCH

    def get_ohlcv_batch(
        self,
        tickers: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        progress: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for many tickers; return dict of {ticker: dataframe}."""
        results = {}
        n = len(tickers)
        for i, t in enumerate(tickers, 1):
            if progress:
                print(f"  [{i:2d}/{n}] {t:6s}", end=' ', flush=True)
            df = self.get_ohlcv(t, start, end)
            if df.empty:
                if progress:
                    print('FAIL')
                continue
            results[t] = df
            if progress:
                print(f'{len(df)} rows')
            # Gentle rate limit
            if i % 10 == 0:
                time.sleep(0.5)
        return results

    # ----------------------------------------------------------- FUNDAMENTALS

    def get_fundamentals(
        self,
        ticker: str,
        max_cache_age_hours: int = 168,  # 7 days
    ) -> dict:
        """Fetch fundamentals snapshot from yfinance .info.

        Returns dict with keys like:
            trailingPE, forwardPE, priceToSalesTrailing12Months,
            earningsQuarterlyGrowth, revenueGrowth, operatingMargins,
            freeCashflow, marketCap, recommendationKey, beta, etc.
        """
        cache_file = self._info_cache / f'{ticker}.json'

        if cache_file.exists():
            with open(cache_file) as f:
                cached = json.load(f)
            age_hours = (datetime.now() - datetime.fromisoformat(cached['fetched_at'])).total_seconds() / 3600
            if age_hours < max_cache_age_hours:
                return cached.get('info', {})

        try:
            ticker_obj = self._yf.Ticker(ticker)
            info = ticker_obj.info
            # Filter to fields we use (yfinance .info has 200+ keys)
            keep = {
                'trailingPE', 'forwardPE', 'priceToSalesTrailing12Months',
                'priceToBook', 'pegRatio', 'enterpriseToRevenue', 'enterpriseToEbitda',
                'earningsQuarterlyGrowth', 'revenueGrowth', 'earningsGrowth',
                'operatingMargins', 'profitMargins', 'grossMargins',
                'returnOnAssets', 'returnOnEquity',
                'freeCashflow', 'operatingCashflow', 'totalCashPerShare',
                'totalRevenue', 'marketCap', 'sharesOutstanding',
                'beta', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow',
                'fiftyDayAverage', 'twoHundredDayAverage',
                'shortRatio', 'shortPercentOfFloat',
                'recommendationKey', 'recommendationMean', 'numberOfAnalystOpinions',
                'targetMeanPrice', 'targetHighPrice', 'targetLowPrice',
                'sector', 'industry',
            }
            filtered = {k: info.get(k) for k in keep}
            with open(cache_file, 'w') as f:
                json.dump({
                    'ticker': ticker,
                    'fetched_at': datetime.now().isoformat(),
                    'info': filtered,
                }, f, default=str)
            return filtered
        except Exception as e:
            logger.error(f"Fundamentals fetch failed for {ticker}: {e}")
            return {}

    # ------------------------------------------------------------ EARNINGS

    def get_next_earnings_date(self, ticker: str) -> Optional[pd.Timestamp]:
        """Best-effort next earnings date from yfinance calendar."""
        try:
            ticker_obj = self._yf.Ticker(ticker)
            cal = ticker_obj.calendar
            if cal is None:
                return None
            if isinstance(cal, dict):
                ed = cal.get('Earnings Date')
                if isinstance(ed, (list, tuple)) and ed:
                    return pd.Timestamp(ed[0])
                if ed is not None:
                    return pd.Timestamp(ed)
            elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.columns:
                return pd.Timestamp(cal['Earnings Date'].iloc[0])
        except Exception:
            pass
        return None

    def get_options_iv(self, ticker: str) -> dict:
        """Pull ATM implied volatility from the nearest expiry options chain.

        Returns dict with at_the_money_iv (mean of nearest ATM call+put IV).
        Returns empty dict if options data unavailable.
        """
        try:
            ticker_obj = self._yf.Ticker(ticker)
            expirations = ticker_obj.options
            if not expirations:
                return {}
            # Use the nearest expiry
            expiry = expirations[0]
            chain = ticker_obj.option_chain(expiry)
            spot = ticker_obj.history(period='1d')['Close'].iloc[-1]
            # ATM strikes for call and put
            calls = chain.calls
            puts = chain.puts
            if calls.empty or puts.empty:
                return {}
            atm_call = calls.iloc[(calls['strike'] - spot).abs().argsort()[:1]]
            atm_put = puts.iloc[(puts['strike'] - spot).abs().argsort()[:1]]
            iv = (atm_call['impliedVolatility'].iloc[0] + atm_put['impliedVolatility'].iloc[0]) / 2
            return {
                'expiry': expiry,
                'spot': spot,
                'atm_iv': iv,
                'days_to_expiry': (pd.Timestamp(expiry) - pd.Timestamp.now()).days,
            }
        except Exception as e:
            logger.debug(f"Options IV fetch failed for {ticker}: {e}")
            return {}

    def clear_cache(self):
        """Remove all cached data."""
        for f in self._ohlcv_cache.glob('*'):
            f.unlink()
        for f in self._info_cache.glob('*'):
            f.unlink()
        logger.info("Cache cleared.")
