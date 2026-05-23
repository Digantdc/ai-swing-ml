"""Transaction cost and slippage model.

Applied to every buy and sell. Designed to reflect realistic retail IBKR costs:
- Commission ~0.05% (negligible for IBKR Pro)
- Effective spread cost ~5-15 bps (varies by liquidity tier)
- Slippage ~5-10 bps on market orders / wide-spread limits
- Plus 'live drag' for any modeled-vs-reality gap

Total per-leg cost = cost_per_leg + slippage + live_drag.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    cost_per_leg: float = 0.0015      # 15 bps base
    slippage: float = 0.0010          # 10 bps
    live_drag: float = 0.0005         # 5 bps for unmodeled friction

    def total_per_leg(self) -> float:
        return self.cost_per_leg + self.slippage + self.live_drag

    def round_trip_cost(self) -> float:
        """Buy + sell = 2x per-leg cost."""
        return 2 * self.total_per_leg()

    def apply_to_return(self, gross_return: float) -> float:
        """Apply round-trip cost to a gross return."""
        return gross_return - self.round_trip_cost()


@dataclass
class OptionsCostModel:
    """Options have wider spreads — apply tier-specific multipliers."""
    base_cost_per_leg: float = 0.005      # 50 bps base
    tier_multiplier: dict[int, float] = None

    def __post_init__(self):
        if self.tier_multiplier is None:
            self.tier_multiplier = {1: 1.0, 2: 1.5, 3: 2.5, 4: 5.0}

    def cost_for_tier(self, tier: int, n_legs: int = 2) -> float:
        """Cost as fraction of trade notional (premium for debit / max-risk for credit)."""
        mult = self.tier_multiplier.get(tier, 2.0)
        return self.base_cost_per_leg * mult * n_legs
