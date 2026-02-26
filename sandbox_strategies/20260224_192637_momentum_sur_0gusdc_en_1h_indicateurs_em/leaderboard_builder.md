# Leaderboard Builder - session 20260224_192637_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : croisement HAUSSE de EMA sur BOLLINGER avec confirmation HAUT du VOLUME_OSCILLATOR, filtré par regime trend via SUPERTREND. Sorties : croisement BAISSSE de EMA ou perte de 20% du spread BOLLINGER. Risk management : SL dynamique basé sur ATR, TP fixe à 2x le SL, avec adaptation de la taille du trade selon le VOLUME_OSCILLATOR. Hypothèse : les mouvements de tendance suivent les impulsions de volume sur les supports/tests de BOLLINGER, mais uniquement dans un contexte de tendance confirmé par SUPERTREND.
Status: failed
Best Sharpe: 0.327
Best Continuous Score: -21.07

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | -21.07 | 0.327 | -3.09% | -44.59% | 0.98 | 61 | continue | needs_work |
| 2 | 2 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 4 | 1 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 5 | 3 | -100.00 | -20.000 | -93.92% | -100.00% | 0.79 | 214 | continue | ruined |