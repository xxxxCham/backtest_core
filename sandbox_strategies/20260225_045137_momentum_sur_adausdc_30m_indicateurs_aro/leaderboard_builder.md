# Leaderboard Builder - session 20260225_045137_momentum_sur_adausdc_30m_indicateurs_aro

Objective: Momentum sur ADAUSDC 30m. Indicateurs : Aroon + EMA + Volume Oscillator. Entrées : Aroon Up cross Aroon Down confirmé par un crossover haussier du Volume Oscillator au-dessus de sa SMA 20, filtré par une volatilité implicite inférieure au 20ème percentile historique. Sorties : EMA croise à la baisse, avec un vote majoritaire de 60% des sessions de trading dans l'heure suivant l'entrée. Risk management : SL adaptatif basé sur l'ATR multiplié par le mode_risk_rotation actuel (0.5x si volatilite faible, 1.5x si volatilite élevée).
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -2.772 | -89.52% | -92.18% | 0.53 | 150 | continue | ruined |
| 2 | 3 | -100.00 | -20.000 | -151.44% | -100.00% | 0.65 | 544 | continue | ruined |
| 3 | 4 | -100.00 | -1.036 | -50.85% | -67.16% | 0.82 | 258 | continue | high_drawdown |