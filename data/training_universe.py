"""Training universe — broader US large-cap basket for model training.

Rationale:
  v1 trained on 60 AI-correlated stocks. Result: model learned "AI sector
  goes up" and stopped. Negative Rank IC.

  v2 trains on ~150 names spanning 7 sectors so the model sees:
    - What a value stock looks like vs a growth stock
    - How defensives behave in risk-off vs cyclicals in risk-on
    - That high-vol AI semis aren't the only pattern in the data

  Prediction is still filtered to the 60 AI/semi/auto trading universe.
  We train wide, trade narrow.
"""
from __future__ import annotations

from .universe import UNIVERSE as TRADING_UNIVERSE

# ---------- Sectors we ADD beyond the trading universe ----------
# Format: ticker -> (sector_label, options_liquidity_tier)
# Tier 1 = penny-spread liquid, Tier 4 = no options. Most diversifiers are Tier 1-2.

TRAINING_ONLY = {
    # ===== Mega-cap tech (different from chip makers) =====
    'AAPL':  ('mega_tech', 1),

    # ===== Financials (rate-sensitive, anti-correlated with growth tech) =====
    'JPM':   ('financials', 1),
    'GS':    ('financials', 1),
    'BAC':   ('financials', 1),
    'MS':    ('financials', 1),
    'C':     ('financials', 1),
    'WFC':   ('financials', 1),
    'SCHW':  ('financials', 1),
    'BLK':   ('financials', 2),
    'AXP':   ('financials', 2),
    'BX':    ('financials', 2),
    'KKR':   ('financials', 2),

    # ===== Energy (commodity-driven, uncorrelated with AI) =====
    'XOM':   ('energy', 1),
    'CVX':   ('energy', 1),
    'COP':   ('energy', 2),
    'SLB':   ('energy', 2),
    'EOG':   ('energy', 2),
    'MPC':   ('energy', 2),
    'PSX':   ('energy', 2),
    'OXY':   ('energy', 2),

    # ===== Healthcare (defensive, anti-cyclical) =====
    'UNH':   ('healthcare', 1),
    'JNJ':   ('healthcare', 1),
    'LLY':   ('healthcare', 1),
    'MRK':   ('healthcare', 1),
    'PFE':   ('healthcare', 1),
    'ABBV':  ('healthcare', 1),
    'TMO':   ('healthcare', 2),
    'ABT':   ('healthcare', 2),
    'BMY':   ('healthcare', 2),
    'CVS':   ('healthcare', 2),
    'DHR':   ('healthcare', 2),
    'GILD':  ('healthcare', 2),

    # ===== Consumer (different volatility regime) =====
    'WMT':   ('consumer', 1),
    'COST':  ('consumer', 1),
    'HD':    ('consumer', 1),
    'NKE':   ('consumer', 2),
    'MCD':   ('consumer', 1),
    'SBUX':  ('consumer', 2),
    'PG':    ('consumer', 1),
    'KO':    ('consumer', 1),
    'PEP':   ('consumer', 1),
    'TGT':   ('consumer', 2),
    'LOW':   ('consumer', 2),
    'DIS':   ('consumer', 2),

    # ===== Industrials (cyclical, different beta) =====
    'CAT':   ('industrials', 1),
    'BA':    ('industrials', 1),
    'GE':    ('industrials', 1),
    'HON':   ('industrials', 2),
    'UPS':   ('industrials', 2),
    'DE':    ('industrials', 2),
    'RTX':   ('industrials', 2),
    'LMT':   ('industrials', 2),
    'NOC':   ('industrials', 2),
    'CSX':   ('industrials', 2),
    'UNP':   ('industrials', 2),

    # ===== Utilities (defensive, low-vol, high yield — opposite of AI) =====
    'NEE':   ('utilities', 2),
    'DUK':   ('utilities', 2),
    'SO':    ('utilities', 2),
    'AEP':   ('utilities', 3),
    'D':     ('utilities', 3),
    'SRE':   ('utilities', 3),

    # ===== Real estate (rate-sensitive, separate factor exposure) =====
    'PLD':   ('real_estate', 2),
    'AMT':   ('real_estate', 2),
    'EQIX':  ('real_estate', 2),
    'CCI':   ('real_estate', 2),

    # ===== Communications (mixed cyclical/defensive) =====
    'NFLX':  ('communications', 1),
    'CMCSA': ('communications', 2),
    'T':     ('communications', 1),
    'VZ':    ('communications', 1),
    'CHTR':  ('communications', 2),

    # ===== Materials =====
    'LIN':   ('materials', 2),
    'FCX':   ('materials', 2),
    'NEM':   ('materials', 2),
    'NUE':   ('materials', 2),

    # ===== Transportation (cyclical) =====
    'FDX':   ('transports', 2),
    'DAL':   ('transports', 2),
    'UAL':   ('transports', 2),
    'LUV':   ('transports', 2),
}

# ----- Combined training universe = TRADING_UNIVERSE + TRAINING_ONLY -----
TRAINING_UNIVERSE: dict[str, tuple[str, int]] = {
    **{t: v for t, v in TRADING_UNIVERSE.items()},
    **TRAINING_ONLY,
}


def get_training_tickers() -> list[str]:
    """Full ~150-name training universe."""
    return list(TRAINING_UNIVERSE.keys())


def get_trading_tickers() -> list[str]:
    """60-name trading universe — predictions are filtered to this list."""
    return list(TRADING_UNIVERSE.keys())


def get_training_sector_map() -> dict[str, str]:
    return {t: v[0] for t, v in TRAINING_UNIVERSE.items()}


def is_in_trading_universe(ticker: str) -> bool:
    return ticker.upper() in TRADING_UNIVERSE
