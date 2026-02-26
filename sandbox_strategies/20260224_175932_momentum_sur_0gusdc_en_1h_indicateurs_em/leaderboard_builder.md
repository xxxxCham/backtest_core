# Leaderboard Builder - session 20260224_175932_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : croisement haussier de EMA 50/200 avec confirmation haute sur BOLLINGER en dessous de la moyenne, volume oscillateur positif. Sorties : croisement baissier EMA avec perte de confiance sur BOLLINGER, ou seuil de volatilité atteint. Risk management : SL basé sur ATR, TP dynamique selon la déviation standard, filtre horaire de liquidité sur les 200 dernières heures. Hypothèse : les mouvements de prix suivant un creux de volatilité avec un volume oscillateur positif révèlent un potentiel de continuation haussier temporaire, testable par backtest sur 6 mois.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |