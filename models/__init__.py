"""ML models package.

Note: heavy ML dependencies (lightgbm, scikit-learn) are imported only
when their respective modules are loaded. Import the symbol you need:

    from models.ranker import LightGBMRanker          # needs lightgbm
    from models.volatility_model import VolatilityModel  # needs scikit-learn
    from models.strategy_selector import StrategySelector  # pure-Python
"""
