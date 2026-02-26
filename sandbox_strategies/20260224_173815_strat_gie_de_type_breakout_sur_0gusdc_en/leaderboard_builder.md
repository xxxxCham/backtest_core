# Leaderboard Builder - session 20260224_173815_strat_gie_de_type_breakout_sur_0gusdc_en

Objective: Stratégie de type breakout sur 0GUSDC en 1h avec filtrage par volatilité. Indicateurs : ATR + KELTNER + VOLUME_OSCILLATOR. Entrées : cassure haute de Keltner avec volume supérieur à la moyenne mobile sur 1h, confirmation par ATR inférieur à la médiane de la volatilité sur 7 jours. Sorties : retour sous la bande de Keltner ou perte de 1.5 ATR. Risk management : SL dynamique basé sur l'ATR, TP fixe à 2x le SL, avec rotation du profil risque selon le niveau de volatilité implicite (mode risque serré si > 75%, mode large si < 25%).
Status: success
Best Sharpe: 1.713
Best Continuous Score: 100.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 100.00 | 1.713 | +36.78% | -11.42% | 1.96 | 26 | accept | target_reached |
| 2 | 1 | 9.42 | 0.279 | +1.88% | -7.50% | 1.09 | 13 | continue | needs_work |