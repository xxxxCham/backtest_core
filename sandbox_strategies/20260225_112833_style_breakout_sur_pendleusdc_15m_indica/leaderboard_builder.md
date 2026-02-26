# Leaderboard Builder - session 20260225_112833_style_breakout_sur_pendleusdc_15m_indica

Objective: Style breakout sur PENDLEUSDC 15m. Indicateurs : EMA + OBV + VOLUME_OSCILLATOR + ATR. Entrées : cassure au-dessus de l'EMA avec confirmation OBV haussière et volume_oscillator croissant. Sorties : retour sous EMA avec divergence OBV baissière. Risk management : SL à 1.5x ATR, TP à 2x ATR, filtre horaire de liquidité entre et. Mode_inverse appliqué : test de logique inverse sur les paires d'indicateurs rarement combinées (OBV + VOLUME_OSCILLATOR). Mode_offbeat : priorisation d'une combinaison non conventionnelle entre OBV et volume_oscillator pour détecter les mouvements de volume non alignés avec le prix. Hypothèse testable : les mouvements de volume non alignés avec le prix indiquent une tendance émergente.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | -100.00 | -20.000 | -324.24% | -100.00% | 0.70 | 1209 | continue | ruined |
| 2 | 3 | -100.00 | -20.000 | -2407.95% | -100.00% | 0.63 | 8232 | continue | ruined |
| 3 | 4 | -100.00 | -20.000 | -877.03% | -100.00% | 0.64 | 3295 | continue | ruined |
| 4 | 5 | -100.00 | -20.000 | -1062.08% | -100.00% | 0.61 | 3860 | continue | ruined |
| 5 | 6 | -100.00 | -20.000 | -1062.08% | -100.00% | 0.61 | 3860 | stop | ruined |