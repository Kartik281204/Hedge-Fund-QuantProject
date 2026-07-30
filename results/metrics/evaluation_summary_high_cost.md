# Evaluation summary

`3000` paired paths per regime, seed `123`. Lower `std_pnl` = less hedging risk. `cvar_5` is the mean P&L of the worst 5% of outcomes (less negative = milder tail losses). `mean_cost` includes the final unwind.

## gbm_high_vol

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | -0.2804 | 1.2450 | -3.4682 | 0.3783 | 1.850 | -0.225 |
| static_bs | -0.4849 | 1.5881 | -4.6230 | 0.5834 | 2.873 | -0.305 |
| adaptive_bs | -0.4248 | 1.0845 | -2.8902 | 0.5248 | 2.580 | -0.392 |

## gbm_low_vol

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | -0.2157 | 0.6965 | -1.7205 | 0.3198 | 1.588 | -0.310 |
| static_bs | -0.3598 | 0.3791 | -1.1162 | 0.4532 | 2.250 | -0.949 |
| adaptive_bs | -0.4366 | 0.4506 | -1.4906 | 0.5279 | 2.624 | -0.969 |

## heston

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | -0.2224 | 0.8567 | -2.1248 | 0.3367 | 1.667 | -0.260 |
| static_bs | -0.3867 | 0.5570 | -1.8267 | 0.4954 | 2.455 | -0.694 |
| adaptive_bs | -0.4310 | 0.6485 | -2.0003 | 0.5313 | 2.636 | -0.665 |

