"""AI ecosystem universe with sector and options liquidity tier metadata."""

# Format: ticker -> (sector, liquidity_tier)
# Liquidity tier: 1 = very liquid (penny spreads), 4 = no/illiquid options.
# Sectors are coarse — used for cross-sectional relative-strength features.

UNIVERSE = {
    # ----- Layer 1: Raw materials & rare earths -----
    'MP':    ('materials', 3),
    'USAR':  ('materials', 4),
    'ALB':   ('materials', 3),
    'TMC':   ('materials', 4),

    # ----- Layer 2: Semi cap eq & EDA -----
    'ASML':  ('semi_cap', 2),
    'AMAT':  ('semi_cap', 2),
    'LRCX':  ('semi_cap', 2),
    'KLAC':  ('semi_cap', 2),
    'TER':   ('semi_cap', 3),
    'CDNS':  ('semi_eda', 3),
    'SNPS':  ('semi_eda', 3),
    'ARM':   ('semi_eda', 2),
    'ACMR':  ('semi_cap', 3),
    'FORM':  ('semi_cap', 3),

    # ----- Layer 3: Foundries & IDMs -----
    'TSM':   ('foundry', 1),
    'INTC':  ('foundry', 1),
    'GFS':   ('foundry', 3),
    'UMC':   ('foundry', 3),
    'TXN':   ('idm', 2),
    'STM':   ('idm', 3),
    'ON':    ('idm', 3),

    # ----- Layer 4: Fabless chip designers -----
    'NVDA':  ('fabless', 1),
    'AMD':   ('fabless', 1),
    'AVGO':  ('fabless', 1),
    'MRVL':  ('fabless', 2),
    'QCOM':  ('fabless', 1),
    'MU':    ('fabless', 1),
    'MBLY':  ('fabless', 2),
    'AMBA':  ('fabless', 3),

    # ----- Layer 5: OSAT / Optical -----
    'AMKR':  ('osat', 3),
    'KLIC':  ('osat', 3),
    'IPGP':  ('optical', 3),
    'COHR':  ('optical', 3),

    # ----- Layer 6: Hardware integrators & data center -----
    'DELL':  ('hw_dc', 2),
    'SMCI':  ('hw_dc', 2),
    'HPE':   ('hw_dc', 2),
    'ANET':  ('hw_dc', 2),
    'CIEN':  ('hw_dc', 3),
    'VRT':   ('hw_dc', 2),
    'ETN':   ('hw_dc', 2),
    'GEV':   ('hw_dc', 2),
    'PWR':   ('hw_dc', 3),
    'NXT':   ('hw_dc', 3),
    'CRWV':  ('hw_dc', 2),
    'NBIS':  ('hw_dc', 3),

    # ----- Layer 7: Autonomous & sensors -----
    'TSLA':  ('autonomous', 1),
    'GOOGL': ('hyperscaler', 1),  # primary listed under hyperscalers but Waymo here
    'HSAI':  ('autonomous', 3),
    'LAZR':  ('autonomous', 3),
    'OUST':  ('autonomous', 3),
    'INVZ':  ('autonomous', 3),
    'AUR':   ('autonomous', 3),
    'APTV':  ('autonomous', 3),
    'XPEV':  ('autonomous', 3),
    'NIO':   ('autonomous', 3),
    'ISRG':  ('autonomous', 3),  # surgical robotics
    'SYM':   ('autonomous', 3),  # warehouse
    'ZBRA':  ('autonomous', 3),
    'SERV':  ('autonomous', 4),

    # ----- Layer 8: Hyperscalers & AI software -----
    'MSFT':  ('hyperscaler', 1),
    'AMZN':  ('hyperscaler', 1),
    'META':  ('hyperscaler', 1),
    'ORCL':  ('hyperscaler', 1),
    'CRM':   ('software', 2),
    'PLTR':  ('software', 1),
    'NOW':   ('software', 2),
    'SNOW':  ('software', 2),
    'DDOG':  ('software', 2),
    'IBM':   ('hyperscaler', 2),
}


def get_sector(ticker: str) -> str:
    """Return sector label for a ticker, or 'unknown' if not in universe."""
    entry = UNIVERSE.get(ticker.upper())
    return entry[0] if entry else 'unknown'


def get_liquidity_tier(ticker: str) -> int:
    """Return options liquidity tier (1-4), or 4 if unknown."""
    entry = UNIVERSE.get(ticker.upper())
    return entry[1] if entry else 4


def get_all_tickers() -> list[str]:
    """Return the full list of tickers in the universe."""
    return list(UNIVERSE.keys())


def get_sectors() -> set[str]:
    """Return the set of all sector labels."""
    return {v[0] for v in UNIVERSE.values()}


def filter_by_tier(max_tier: int = 3) -> list[str]:
    """Return tickers with liquidity tier <= max_tier (default: exclude Tier 4)."""
    return [t for t, (_, tier) in UNIVERSE.items() if tier <= max_tier]
