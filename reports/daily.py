"""Generate daily prediction reports and backtest summaries as markdown."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def write_daily_report(
    picks: pd.DataFrame,
    options_trades: list[dict],
    out_path: Path | str,
    asof_date: pd.Timestamp,
    regime_info: dict | None = None,
    portfolio_stats: dict | None = None,
) -> Path:
    """Write today's top-N picks and options recommendations.

    Args:
        picks: DataFrame ['ticker', 'spot', 'score', 'score_pct', 'sector',
                          'predicted_vol', 'liquidity_tier']
        options_trades: list of OptionsTrade.to_dict() outputs
        out_path: output directory
        asof_date: date these picks are for

    Returns:
        Path to the written markdown file.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f'daily_picks_{asof_date.strftime("%Y-%m-%d")}.md'

    lines = []
    lines.append(f"# AI Swing ML — Daily Picks {asof_date.strftime('%Y-%m-%d')}")
    lines.append(f"\n_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    # Regime header
    if regime_info:
        lines.append(f"**Market regime:** `{regime_info.get('regime', 'unknown')}` "
                     f"(score {regime_info.get('regime_score', 0):+.2f}) · "
                     f"VIX {regime_info.get('vix', 0) or 0:.1f} · "
                     f"SPY {'above' if regime_info.get('spy_above_200ma') else 'below'} 200-SMA · "
                     f"21d return {regime_info.get('spy_21d_ret', 0) or 0:+.2%}\n")
    if portfolio_stats:
        lines.append(f"**Portfolio sizing:** {portfolio_stats.get('n_positions', 0)} positions · "
                     f"gross {portfolio_stats.get('gross_exposure', 0):.0%} · "
                     f"expected Sharpe {portfolio_stats.get('expected_sharpe', 0):.2f} · "
                     f"21d expected return {portfolio_stats.get('expected_return_21d', 0):+.2%}\n")

    lines.append("## Top picks (ranked by model score)\n")
    if picks.empty:
        lines.append("_No picks generated today._\n")
    else:
        has_kelly = 'kelly_weight' in picks.columns
        if has_kelly:
            lines.append("| Rank | Ticker | Spot | Score | Pct | Sector | Pred Vol | Tier | Weight |")
            lines.append("|---|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| Rank | Ticker | Spot | Score | Pct | Sector | Pred Vol | Tier |")
            lines.append("|---|---|---|---|---|---|---|---|")
        for i, row in picks.iterrows():
            base = (
                f"| {i+1} | **{row['ticker']}** | ${row.get('spot', 0):.2f} "
                f"| {row['score']:.3f} | {row.get('score_pct', 0):.0%} "
                f"| {row.get('sector', '-')} | "
                f"{row.get('predicted_vol', 0):.0%} | {row.get('liquidity_tier', '-')}"
            )
            if has_kelly:
                base += f" | {row.get('kelly_weight', 0):.1%} |"
            else:
                base += " |"
            lines.append(base)

    lines.append("\n## Options strategy recommendations\n")
    if not options_trades:
        lines.append("_No options trades recommended (insufficient signal or liquidity)._\n")
    else:
        for trade in options_trades:
            if trade.get('strategy') == 'wait':
                continue
            lines.append(f"### {trade['ticker']} — {trade['strategy'].replace('_', ' ').title()}")
            lines.append(f"- **Spot:** ${trade['spot']:.2f} · **DTE:** {trade['expiry_dte']}d")
            lines.append(f"- **Direction score (pct):** {trade.get('direction_score_pct', 0):.2f}")
            lines.append(f"- **Predicted 21d vol:** {trade.get('predicted_vol', 0):.0%}")
            if trade.get('iv_rank') is not None:
                lines.append(f"- **IV rank estimate:** {trade['iv_rank']:.0f}")
            lines.append("- **Legs:**")
            for leg in trade['legs']:
                lines.append(
                    f"  - {leg['action'].upper()} {leg['qty']}× {leg['type'].upper()} "
                    f"@ ${leg['strike']:.2f} (delta ~{leg.get('delta', 0):.2f})"
                )
            lines.append(f"- **Max gain:** ${trade['max_gain']:,.0f} · **Max loss:** ${trade['max_loss']:,.0f}")
            lines.append(f"- **Breakeven:** ${trade['breakeven']:.2f} · **POP est:** {trade['pop_estimate']:.0%}")
            lines.append(f"- **Rationale:** {trade['rationale']}\n")

    lines.append("\n---\n_Research only, not investment advice. Model outputs are statistical signals, not certainties._")

    fname.write_text('\n'.join(lines))
    return fname


def write_backtest_report(
    metrics: dict,
    feature_importance: pd.DataFrame,
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    options_pnl: pd.DataFrame | None,
    out_path: Path | str,
) -> Path:
    """Write backtest summary to markdown."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / 'backtest_report.md'

    lines = []
    lines.append("# Backtest Report\n")
    lines.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    lines.append("## Equity-leg performance\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k in ['total_return', 'annualized_return', 'annualized_volatility',
              'sharpe', 'sortino', 'max_drawdown', 'calmar',
              'hit_rate_days', 'n_trades', 'hit_rate_trades',
              'avg_winner', 'avg_loser', 'profit_factor',
              'information_coefficient', 'rank_ic']:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, float):
            if 'return' in k or 'drawdown' in k or 'rate' in k or 'winner' in k or 'loser' in k:
                lines.append(f"| {k.replace('_', ' ').title()} | {v:.2%} |")
            else:
                lines.append(f"| {k.replace('_', ' ').title()} | {v:.3f} |")
        else:
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

    lines.append("\n## How to interpret these numbers\n")
    sharpe = metrics.get('sharpe', 0)
    if sharpe > 1.5:
        lines.append(f"- **Sharpe {sharpe:.2f}** is good. Verify it survives out-of-sample.")
    elif sharpe > 0.8:
        lines.append(f"- **Sharpe {sharpe:.2f}** is decent for a retail strategy.")
    elif sharpe > 0:
        lines.append(f"- **Sharpe {sharpe:.2f}** is positive but weak — re-tune or simplify.")
    else:
        lines.append(f"- **Sharpe {sharpe:.2f}** is negative — the model is not better than cash. Do not trade.")

    rank_ic = metrics.get('rank_ic', 0)
    if rank_ic > 0.05:
        lines.append(f"- **Rank IC {rank_ic:.3f}** shows real predictive power.")
    elif rank_ic > 0:
        lines.append(f"- **Rank IC {rank_ic:.3f}** is small. Edge is fragile.")
    else:
        lines.append(f"- **Rank IC {rank_ic:.3f}** ≤ 0. No predictive power found.")

    mdd = metrics.get('max_drawdown', 0)
    if abs(mdd) > 0.30:
        lines.append(f"- **Max drawdown {mdd:.2%}** is severe. Consider tighter stops or smaller positions.")

    lines.append("\n## Top 20 features by importance\n")
    if not feature_importance.empty:
        top20 = feature_importance.head(20)
        lines.append("| Feature | Importance |")
        lines.append("|---|---|")
        for _, row in top20.iterrows():
            lines.append(f"| `{row['feature']}` | {row['importance']:.0f} |")

    # Options overlay
    if options_pnl is not None and not options_pnl.empty:
        lines.append("\n## Options overlay summary\n")
        net = options_pnl['net_pnl'].sum()
        n = len(options_pnl)
        winners = (options_pnl['net_pnl'] > 0).sum()
        lines.append(f"- **Trades:** {n}")
        lines.append(f"- **Win rate:** {winners/max(n,1):.1%}")
        lines.append(f"- **Total net P&L:** ${net:,.0f}")
        lines.append(f"- **Avg per trade:** ${options_pnl['net_pnl'].mean():,.0f}")
        lines.append("\n### By strategy\n")
        by_strat = options_pnl.groupby('strategy').agg(
            n=('net_pnl', 'count'),
            total_pnl=('net_pnl', 'sum'),
            avg_pnl=('net_pnl', 'mean'),
            win_rate=('net_pnl', lambda x: (x > 0).mean()),
        ).reset_index()
        lines.append("| Strategy | N | Total P&L | Avg P&L | Win rate |")
        lines.append("|---|---|---|---|---|")
        for _, row in by_strat.iterrows():
            lines.append(
                f"| {row['strategy']} | {row['n']} | ${row['total_pnl']:,.0f} "
                f"| ${row['avg_pnl']:,.0f} | {row['win_rate']:.1%} |"
            )

    lines.append("\n---\n_Backtest results are not a guarantee of future performance. "
                 "Run multiple seeds, multiple sub-periods, and check for regime sensitivity._")

    fname.write_text('\n'.join(lines))

    # Also save CSVs
    if not equity_curve.empty:
        equity_curve.to_csv(out_path / 'backtest_equity.csv', index=False)
    if not trades.empty:
        trades.to_csv(out_path / 'backtest_trades.csv', index=False)
    if not feature_importance.empty:
        feature_importance.to_csv(out_path / 'feature_importance.csv', index=False)
    if options_pnl is not None and not options_pnl.empty:
        options_pnl.to_csv(out_path / 'backtest_options.csv', index=False)

    return fname
