# Evaluation summary

`3000` paired paths per regime, seed `123`. Lower `std_pnl` = less hedging risk. `cvar_5` is the mean P&L of the worst 5% of outcomes (less negative = milder tail losses). `mean_cost` includes the final unwind.

## gbm_high_vol

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | 0.0255 | 1.6515 | -4.5558 | 0.0965 | 1.893 | 0.015 |
| static_bs | -0.0473 | 1.4747 | -3.9518 | 0.1459 | 2.873 | -0.032 |
| adaptive_bs | -0.0312 | 1.0600 | -2.4440 | 0.1312 | 2.580 | -0.029 |

## gbm_low_vol

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | 0.0332 | 0.7301 | -1.8352 | 0.0839 | 1.667 | 0.045 |
| static_bs | -0.0200 | 0.3744 | -0.7329 | 0.1133 | 2.250 | -0.053 |
| adaptive_bs | -0.0406 | 0.4174 | -1.0136 | 0.1320 | 2.624 | -0.097 |

## heston

| method | mean_pnl | std_pnl | cvar_5 | mean_cost | turnover | sharpe-like |
|---|---:|---:|---:|---:|---:|---:|
| rl_ppo | 0.0246 | 1.0373 | -2.5812 | 0.0881 | 1.746 | 0.024 |
| static_bs | -0.0152 | 0.5333 | -1.3349 | 0.1238 | 2.455 | -0.028 |
| adaptive_bs | -0.0326 | 0.6297 | -1.5593 | 0.1328 | 2.636 | -0.052 |

