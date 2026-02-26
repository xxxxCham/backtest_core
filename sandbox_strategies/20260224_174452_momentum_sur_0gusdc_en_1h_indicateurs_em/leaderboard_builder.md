# Leaderboard Builder - session 20260224_174452_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : croisement haut de EMA vers le haut avec prix qui sort de bande de Bollinger vers le haut, confirmé par volume oscillant positif. Sorties : croisement bas de EMA vers le bas ou retour dans bande de Bollinger. Risk management : SL à 1.5x ATR, TP à 2x ATR avec ajustement dynamique selon volatilité réelle. Filtre horaire : uniquement les sessions avec volume > médiane de 1h et liquidité > 500k.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -51.79 | -0.467 | -18.70% | -33.25% | 0.80 | 67 | continue | losing_per_trade |
| 2 | 3 | -100.00 | -20.000 | -130.30% | -100.00% | 0.73 | 304 | stop | ruined |