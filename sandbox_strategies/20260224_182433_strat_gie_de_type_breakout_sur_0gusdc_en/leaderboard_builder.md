# Leaderboard Builder - session 20260224_182433_strat_gie_de_type_breakout_sur_0gusdc_en

Objective: Stratégie de type breakout sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : cassure au-dessus de la bande supérieure de Bollinger avec confirmation EMA 50/200 et volume oscillateur > 0. Sorties : retour sous la bande inférieure avec EMA en divergence baissière ou volume oscillateur < 0. Risk management : SL dynamique basé sur ATR, TP fixe à 2x le SL, filtre horaire 1h-1h pour éviter les sessions faibles en liquidité, et gating par volatilité implicite via PI_CYCLE pour ajuster la taille du trade selon le contexte de marché.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |