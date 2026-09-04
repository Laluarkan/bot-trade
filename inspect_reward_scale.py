"""
Diagnostic: prints the empirical per-bar return statistics of your own data,
and compares them against the reward coefficients that will be used, so you
can sanity-check the scale BEFORE spending hours training.

If |beta*drawdown| or |delta| end up much larger than a typical |return|,
the agent is likely to reward-hack (e.g. always hold one position just to
farm the stability bonus) rather than actually learn to trade -- which is
what happened with the default paper constants on M1 data.

Usage
-----
python inspect_reward_scale.py --symbol XAUUSDm --timeframe 1M --data-dir ./data
"""
import argparse
import numpy as np

import config
from data_loader import load_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=config.DEFAULT_SYMBOL)
    p.add_argument("--timeframe", default=config.DEFAULT_TIMEFRAME)
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    args = p.parse_args()

    df = load_csv(args.symbol, args.timeframe, data_dir=args.data_dir)
    close = df["close"].to_numpy(dtype=np.float64)
    returns = np.diff(close) / close[:-1]

    print(f"\n=== Per-bar return statistics: {args.symbol}_{args.timeframe} ===")
    print(f"n bars              : {len(close):,}")
    print(f"mean |return|        : {np.abs(returns).mean():.6f}  ({np.abs(returns).mean()*100:.4f}%)")
    print(f"return std (1 bar)   : {returns.std():.6f}  ({returns.std()*100:.4f}%)")

    bars_per_day = max(1, int(24 * 60 / config.TIMEFRAME_MINUTES.get(args.timeframe, 60)))
    daily_equiv_std = returns.std() * np.sqrt(bars_per_day)
    print(f"return std (~1 day, scaled)          : {daily_equiv_std:.6f} ({daily_equiv_std*100:.4f}%)")

    c = config.reward_coeffs_for(args.timeframe)
    print(f"\nReward coefficients for '{args.timeframe}': {c}")
    print("\nRough scale comparison (rule of thumb: these should be roughly")
    print("the same order of magnitude as 'mean |return|' above, not 100x+ bigger):")
    print(f"  alpha * mean|return|            = {c['alpha'] * np.abs(returns).mean():.6f}")
    print(f"  delta (flat stability bonus)    = {c['delta']:.6f}")
    print(f"  beta  * 1% drawdown             = {c['beta'] * 0.01:.6f}")
    print(f"  gamma * 1 round-trip cost (~0.02%) = {c['gamma'] * 0.0002:.6f}")

    ratio = c["delta"] / max(np.abs(returns).mean(), 1e-12)
    print(f"\ndelta / mean|return| ratio = {ratio:.2f}x")
    if ratio > 5:
        print("WARNING: delta is more than 5x a typical bar's return. The agent can likely")
        print("earn more reward by NEVER changing position than by actually trading well.")
        print("Consider lowering REWARD_COEFFS_BY_TIMEFRAME[...]['delta'] in config.py.")
    else:
        print("OK: delta looks reasonably scaled relative to typical per-bar returns.")


if __name__ == "__main__":
    main()