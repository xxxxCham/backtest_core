# Leaderboard Builder - session 20260225_054406_momentum_sur_atomusdc_30m_indicateurs_em

Objective: Momentum sur ATOMUSDC 30m. Indicateurs : EMA + MFI + Vortex. Entrées : Croisement haussier de l'EMA (20 périodes) sous l'MFI (14 périodes) confirmé par un croisement haussier du Vortex (22,3) et un vote majoritaire des signaux sur 3 périodes, avec gating par volatilité implicite inférieure à 20%. Sorties : Reversals de tendance confirmés par un rejet de la moyenne mobile à 20 périodes et un volume minimal sur la dernière bougie. Risk management : SL basé sur l'ATR ajusté dynamiquement en fonction du contexte de marché et TP à 1.5x le risque initial, avec un SL adaptatif pour gérer la volatilité.
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -24.35 | -0.005 | -8.21% | -34.54% | 0.95 | 146 | continue | needs_work |
| 2 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 1 | -100.00 | -0.827 | -56.98% | -80.39% | 0.63 | 132 | continue | high_drawdown |
| 4 | 3 | -100.00 | -20.000 | -215.59% | -100.00% | 0.58 | 750 | continue | ruined |