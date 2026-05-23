"""Backtest engine package.

Like models/, heavy submodules are loaded only when their symbols are imported.

    from backtest.costs import CostModel, OptionsCostModel        # pure-Python
    from backtest.metrics import compute_metrics, sharpe_ratio    # pure-Python
    from backtest.walk_forward import WalkForwardBacktest         # pure-Python
    from backtest.portfolio import PortfolioBacktester            # pure-Python
    from backtest.options import OptionsBacktester                # transitively pulls strategy_selector
"""
