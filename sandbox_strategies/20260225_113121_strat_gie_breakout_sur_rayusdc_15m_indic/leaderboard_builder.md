# Leaderboard Builder - session 20260225_113121_strat_gie_breakout_sur_rayusdc_15m_indic

Objective: Stratégie breakout sur RAYUSDC 15m. Indicateurs : EMA + AROON + VOLUME_OSCILLATOR + ATR. Entrées : crossover EMA avec AROON haussier + volume supérieur à la moyenne mobile sur 20 périodes. Sorties : perte de momentum sur 5 périodes avec AROON descendant. Risk management : SL à 1.5x ATR, TP à 2x ATR avec ajustement dynamique selon volatilité récente. Mode_microstructure : filtre sur les heures de forte liquidité (09h-11h et-17h). Mode_risk_rotation : rotation entre SL/TP serrés (volatilité faible) et larges (volatilité élevée). Hypothèse testable : les ruptures avec forte volume et liquidité horaire ont 60% de réussite dans un contexte de tendance forte.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -50.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 5 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 1 | -100.00 | -20.000 | -122.62% | -100.00% | 0.83 | 796 | continue | ruined |
| 4 | 2 | -100.00 | -20.000 | -1171.63% | -100.00% | 0.70 | 4102 | continue | ruined |
| 5 | 3 | -100.00 | -20.000 | -1171.63% | -100.00% | 0.70 | 4102 | continue | ruined |
| 6 | 6 | -100.00 | -20.000 | -1604.43% | -100.00% | 0.55 | 4976 | stop | ruined |