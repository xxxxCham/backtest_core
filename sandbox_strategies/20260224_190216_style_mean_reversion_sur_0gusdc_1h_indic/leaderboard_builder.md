# Leaderboard Builder - session 20260224_190216_style_mean_reversion_sur_0gusdc_1h_indic

Objective: Style : Mean-Reversion sur 0GUSDC 1h. Indicateurs : STOCH_RSI + VORTIC + ATR. Entrées : crossing inférieur à la ligne de 20 du STOCH_RSI en contexte de volatilité haussière confirmé par un ATR croissant, avec un VORTIC négatif sur 2 périodes. Sorties : crossover au-dessus de la ligne de 80 du STOCH_RSI ou perte de la tendance confirmée par le VORTIC. Risk management : SL à 1.5x ATR, TP à 1.5x le retour à la moyenne de prix, avec gating horaire de liquidité entre 08h00 et 16h00 UTC.
Status: failed
Best Sharpe: 0.212
Best Continuous Score: 5.52

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 5.52 | 0.212 | -0.78% | -24.59% | 0.99 | 83 | continue | needs_work |
| 2 | 6 | 5.52 | 0.212 | -0.78% | -24.59% | 0.99 | 83 | stop | needs_work |
| 3 | 4 | -15.15 | 0.036 | -1.27% | -14.11% | 0.96 | 16 | continue | needs_work |
| 4 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |