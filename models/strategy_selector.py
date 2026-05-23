"""v2 rule-based options-strategy selector with 7 strategies.

Maps (direction probability, predicted vol, IV regime, earnings proximity)
to the best-fit options structure.

Strategies supported:
    1. long_call               — strongly bullish + low IV + no earnings
    2. bull_call_spread        — moderately bullish + mid/high IV
    3. bull_put_credit         — moderately bullish + high IV (collect premium)
    4. pmcc                    — bullish bias + mid IV + slow drift expected
    5. short_strangle          — neutral + high IV + range-bound
    6. iron_condor             — neutral + high IV + non-event
    7. earnings_iron_condor    — neutral + earnings within 7 days + HIGH IV
    8. calendar_spread         — neutral + low IV + expecting IV expansion
    9. wait                    — no setup
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptionsTrade:
    ticker: str
    spot: float
    strategy: str
    expiry_dte: int
    legs: list[dict] = field(default_factory=list)
    rationale: str = ''
    max_loss: float = 0.0
    max_gain: float = 0.0
    breakeven: float = 0.0
    pop_estimate: float = 0.0
    iv_rank: Optional[float] = None
    predicted_vol: Optional[float] = None
    direction_score_pct: Optional[float] = None
    days_to_earnings: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            'ticker': self.ticker, 'spot': self.spot, 'strategy': self.strategy,
            'expiry_dte': self.expiry_dte, 'legs': self.legs,
            'rationale': self.rationale,
            'max_loss': self.max_loss, 'max_gain': self.max_gain,
            'breakeven': self.breakeven, 'pop_estimate': self.pop_estimate,
            'iv_rank': self.iv_rank, 'predicted_vol': self.predicted_vol,
            'direction_score_pct': self.direction_score_pct,
            'days_to_earnings': self.days_to_earnings,
        }


class StrategySelector:
    """Map (direction × vol × IV × earnings) → options trade."""

    def __init__(
        self,
        expiry_dte_short: int = 30,
        expiry_dte_pmcc_long: int = 270,
        expiry_dte_calendar_long: int = 75,
        expiry_dte_earnings_ic: int = 21,
        iv_to_rv_ratio: float = 1.20,
        risk_per_trade: float = 0.02,
    ):
        self.expiry_dte_short = expiry_dte_short
        self.expiry_dte_pmcc_long = expiry_dte_pmcc_long
        self.expiry_dte_calendar_long = expiry_dte_calendar_long
        self.expiry_dte_earnings_ic = expiry_dte_earnings_ic
        self.iv_to_rv_ratio = iv_to_rv_ratio
        self.risk_per_trade = risk_per_trade

    def select(
        self,
        ticker: str,
        spot: float,
        direction_prob: float,
        predicted_vol: float,
        current_rv_21d: float | None = None,
        iv_rank: float | None = None,
        liquidity_tier: int = 2,
        days_to_earnings: int | None = None,
        regime_score: float = 0.0,
    ) -> OptionsTrade:
        """
        direction_prob: P(stock_up_3%+_in_21d) from binary classifier (0..1)
        predicted_vol: predicted 21d realized vol (annualized, e.g. 0.40)
        iv_rank: 0..100 percentile, if available
        days_to_earnings: int, or None if unknown / no earnings within window
        regime_score: -1 to +1 from RegimeDetector (gates aggressive plays)
        """
        # Hard skip
        if liquidity_tier >= 4:
            return OptionsTrade(
                ticker=ticker, spot=spot, strategy='wait', expiry_dte=0,
                rationale='Tier 4 options liquidity — use shares only.',
                direction_score_pct=direction_prob, predicted_vol=predicted_vol,
                days_to_earnings=days_to_earnings,
            )

        # Estimate IV regime if not provided
        if iv_rank is None and current_rv_21d is not None and predicted_vol is not None:
            iv_rank = min(100, max(0, 50 + 100 * (predicted_vol / max(current_rv_21d, 0.05) - 1)))

        # Regime gate — risk-off regime kills aggressive plays
        if regime_score < -0.5 and direction_prob < 0.65:
            return OptionsTrade(
                ticker=ticker, spot=spot, strategy='wait', expiry_dte=0,
                rationale=f'Risk-off regime ({regime_score:+.2f}) without high conviction.',
                direction_score_pct=direction_prob, predicted_vol=predicted_vol,
                iv_rank=iv_rank, days_to_earnings=days_to_earnings,
            )

        high_iv = iv_rank is not None and iv_rank >= 60
        mid_iv = iv_rank is not None and 30 <= iv_rank < 60
        low_iv = iv_rank is not None and iv_rank < 30

        strong_bull = direction_prob >= 0.70
        moderate_bull = 0.60 <= direction_prob < 0.70
        neutral = 0.40 <= direction_prob < 0.60
        weak_or_bear = direction_prob < 0.40

        atm = spot
        otm_5 = spot * 1.05
        otm_10 = spot * 1.10
        otm_neg_5 = spot * 0.95
        otm_neg_10 = spot * 0.90
        deep_itm = spot * 0.85
        otm_neg_3 = spot * 0.97
        otm_3 = spot * 1.03
        otm_neg_7 = spot * 0.93
        otm_7 = spot * 1.07

        # ----- EARNINGS PLAYS (priority — earnings dominate IV) -----
        earnings_soon = days_to_earnings is not None and 0 <= days_to_earnings <= 7
        if earnings_soon and high_iv and neutral:
            return self._build_earnings_iron_condor(
                ticker, spot, otm_neg_7, otm_neg_10, otm_7, otm_10,
                direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- STRONG BULL -----
        if strong_bull and low_iv:
            return self._build_long_call(ticker, spot, atm, direction_prob, predicted_vol, iv_rank, days_to_earnings)
        if strong_bull and (mid_iv or high_iv):
            return self._build_bull_call_spread(
                ticker, spot, atm, otm_10, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- MODERATE BULL with longer thesis: PMCC -----
        if moderate_bull and mid_iv and regime_score > 0:
            return self._build_pmcc(
                ticker, spot, deep_itm, otm_5, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- MODERATE BULL with high IV: sell premium -----
        if moderate_bull and high_iv:
            return self._build_bull_put_credit(
                ticker, spot, otm_neg_5, otm_neg_10, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- MODERATE BULL with normal IV -----
        if moderate_bull and not high_iv:
            return self._build_bull_call_spread(
                ticker, spot, atm, otm_5, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- NEUTRAL with high IV: iron condor or short strangle -----
        if neutral and high_iv:
            # For Tier 1-2 liquidity and high vol stock, prefer iron condor (defined risk)
            if liquidity_tier <= 2 and predicted_vol > 0.40:
                return self._build_iron_condor(
                    ticker, spot, otm_neg_5, otm_neg_10, otm_5, otm_10,
                    direction_prob, predicted_vol, iv_rank, days_to_earnings,
                )
            return self._build_short_strangle(
                ticker, spot, otm_neg_7, otm_7, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # ----- NEUTRAL with low IV: calendar spread -----
        if neutral and low_iv:
            return self._build_calendar_spread(
                ticker, spot, atm, direction_prob, predicted_vol, iv_rank, days_to_earnings,
            )

        # Default: wait
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='wait', expiry_dte=0,
            rationale=f'No setup match — direction {direction_prob:.2f}, IV rank {iv_rank}, regime {regime_score:+.2f}',
            direction_score_pct=direction_prob, predicted_vol=predicted_vol,
            iv_rank=iv_rank, days_to_earnings=days_to_earnings,
        )

    # ----------------------------------------------------- strategy builders

    def _build_long_call(self, ticker, spot, strike, dscore, pvol, iv_rank, dte_earn):
        premium = spot * max(0.04, pvol / 8)
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='long_call',
            expiry_dte=self.expiry_dte_short,
            legs=[{'action': 'buy', 'type': 'call', 'strike': round(strike, 2), 'qty': 1, 'delta': 0.55}],
            max_loss=premium * 100, max_gain=float('inf'),
            breakeven=strike + premium, pop_estimate=0.42,
            rationale=f'Strong bull ({dscore:.2f}) + low IV — buy directional premium.',
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_bull_call_spread(self, ticker, spot, long_k, short_k, dscore, pvol, iv_rank, dte_earn):
        width = short_k - long_k
        debit = width * 0.40
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='bull_call_spread',
            expiry_dte=self.expiry_dte_short,
            legs=[
                {'action': 'buy', 'type': 'call', 'strike': round(long_k, 2), 'qty': 1, 'delta': 0.50},
                {'action': 'sell', 'type': 'call', 'strike': round(short_k, 2), 'qty': 1, 'delta': 0.25},
            ],
            max_loss=debit * 100, max_gain=(width - debit) * 100,
            breakeven=long_k + debit, pop_estimate=0.38,
            rationale=f'Bullish ({dscore:.2f}), elevated IV — defined-risk debit spread.',
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_bull_put_credit(self, ticker, spot, short_k, long_k, dscore, pvol, iv_rank, dte_earn):
        width = short_k - long_k
        credit = width * 0.30
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='bull_put_credit',
            expiry_dte=self.expiry_dte_short,
            legs=[
                {'action': 'sell', 'type': 'put', 'strike': round(short_k, 2), 'qty': 1, 'delta': -0.30},
                {'action': 'buy', 'type': 'put', 'strike': round(long_k, 2), 'qty': 1, 'delta': -0.15},
            ],
            max_gain=credit * 100, max_loss=(width - credit) * 100,
            breakeven=short_k - credit, pop_estimate=0.62,
            rationale=f'Bull bias ({dscore:.2f}), high IV — sell premium, collect theta.',
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_pmcc(self, ticker, spot, deep_itm_k, short_k, dscore, pvol, iv_rank, dte_earn):
        # Long leg: deep-ITM ~9 months out, ~0.80 delta
        # Short leg: monthly OTM call, ~0.30 delta
        long_cost = (spot - deep_itm_k + spot * 0.05) * 100  # intrinsic + small time
        short_credit = spot * 0.015 * 100  # ~1.5% premium for monthly OTM call
        net_debit = long_cost - short_credit
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='pmcc',
            expiry_dte=self.expiry_dte_short,  # report short leg DTE
            legs=[
                {'action': 'buy', 'type': 'call', 'strike': round(deep_itm_k, 2), 'qty': 1, 'delta': 0.80,
                 'expiry_dte_leg': self.expiry_dte_pmcc_long},
                {'action': 'sell', 'type': 'call', 'strike': round(short_k, 2), 'qty': 1, 'delta': 0.30,
                 'expiry_dte_leg': self.expiry_dte_short},
            ],
            max_loss=net_debit, max_gain=(short_k - deep_itm_k) * 100 - net_debit,
            breakeven=deep_itm_k + net_debit / 100, pop_estimate=0.55,
            rationale=f'Moderate bull ({dscore:.2f}), normal IV — PMCC as stock substitute. Long LEAP {self.expiry_dte_pmcc_long}d, short monthly.',
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_short_strangle(self, ticker, spot, put_k, call_k, dscore, pvol, iv_rank, dte_earn):
        # UNDEFINED RISK — should only be sized small
        put_premium = spot * 0.012
        call_premium = spot * 0.012
        total_credit = (put_premium + call_premium) * 100
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='short_strangle',
            expiry_dte=self.expiry_dte_short,
            legs=[
                {'action': 'sell', 'type': 'put', 'strike': round(put_k, 2), 'qty': 1, 'delta': -0.25},
                {'action': 'sell', 'type': 'call', 'strike': round(call_k, 2), 'qty': 1, 'delta': 0.25},
            ],
            max_gain=total_credit,
            max_loss=float('inf'),  # UNDEFINED RISK
            breakeven=spot,
            pop_estimate=0.60,
            rationale=(
                f'Neutral ({dscore:.2f}) + high IV — sell range. '
                f'⚠ UNDEFINED RISK. Size small ({self.risk_per_trade*100:.0f}% account max).'
            ),
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_iron_condor(self, ticker, spot, put_short, put_long, call_short, call_long, dscore, pvol, iv_rank, dte_earn):
        put_width = put_short - put_long
        call_width = call_long - call_short
        credit = (put_width + call_width) * 0.25
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='iron_condor',
            expiry_dte=self.expiry_dte_short,
            legs=[
                {'action': 'sell', 'type': 'put', 'strike': round(put_short, 2), 'qty': 1, 'delta': -0.25},
                {'action': 'buy', 'type': 'put', 'strike': round(put_long, 2), 'qty': 1, 'delta': -0.12},
                {'action': 'sell', 'type': 'call', 'strike': round(call_short, 2), 'qty': 1, 'delta': 0.25},
                {'action': 'buy', 'type': 'call', 'strike': round(call_long, 2), 'qty': 1, 'delta': 0.12},
            ],
            max_gain=credit * 100,
            max_loss=(max(put_width, call_width) - credit) * 100,
            breakeven=spot, pop_estimate=0.55,
            rationale=f'Neutral ({dscore:.2f}) + high IV — defined-risk premium collection.',
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_earnings_iron_condor(self, ticker, spot, put_short, put_long, call_short, call_long, dscore, pvol, iv_rank, dte_earn):
        put_width = put_short - put_long
        call_width = call_long - call_short
        # Earnings IV is rich — wider credit assumption
        credit = (put_width + call_width) * 0.30
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='earnings_iron_condor',
            expiry_dte=self.expiry_dte_earnings_ic,
            legs=[
                {'action': 'sell', 'type': 'put', 'strike': round(put_short, 2), 'qty': 1, 'delta': -0.25},
                {'action': 'buy', 'type': 'put', 'strike': round(put_long, 2), 'qty': 1, 'delta': -0.10},
                {'action': 'sell', 'type': 'call', 'strike': round(call_short, 2), 'qty': 1, 'delta': 0.25},
                {'action': 'buy', 'type': 'call', 'strike': round(call_long, 2), 'qty': 1, 'delta': 0.10},
            ],
            max_gain=credit * 100,
            max_loss=(max(put_width, call_width) - credit) * 100,
            breakeven=spot, pop_estimate=0.50,
            rationale=(
                f'Earnings in {dte_earn}d + high IV — pure IV crush play. '
                f'Expiry {self.expiry_dte_earnings_ic}d (just after print).'
            ),
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )

    def _build_calendar_spread(self, ticker, spot, strike, dscore, pvol, iv_rank, dte_earn):
        # Sell short-term ATM, buy longer-term ATM
        short_premium = spot * 0.015 * 100
        long_premium = spot * 0.025 * 100
        net_debit = long_premium - short_premium
        return OptionsTrade(
            ticker=ticker, spot=spot, strategy='calendar_spread',
            expiry_dte=self.expiry_dte_short,
            legs=[
                {'action': 'sell', 'type': 'call', 'strike': round(strike, 2), 'qty': 1, 'delta': 0.50,
                 'expiry_dte_leg': self.expiry_dte_short},
                {'action': 'buy', 'type': 'call', 'strike': round(strike, 2), 'qty': 1, 'delta': 0.50,
                 'expiry_dte_leg': self.expiry_dte_calendar_long},
            ],
            max_loss=net_debit,
            max_gain=net_debit * 1.5,  # rough — peaks at short-leg expiry
            breakeven=strike,
            pop_estimate=0.50,
            rationale=(
                f'Neutral ({dscore:.2f}) + low IV — calendar spread to capture IV expansion. '
                f'Short {self.expiry_dte_short}d, long {self.expiry_dte_calendar_long}d.'
            ),
            iv_rank=iv_rank, predicted_vol=pvol, direction_score_pct=dscore,
            days_to_earnings=dte_earn,
        )
