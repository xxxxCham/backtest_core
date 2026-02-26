# Leaderboard Builder - session 20260224_213949_objectif_de_strat_gie_acheter_sur_tendan

Objective: Objectif de Stratégie:
"Acheter sur tendance avec RSI<30 et Aroon Up dans un marché Crypto (0GUSDC, 1h timeframe) en utilisant la combination EMA + OBV pour les entrées. Supprimer les transactions où le volume est faible ou si des signaux contradictoires sont présents avec une majorité de votes pour l'un d'entre eux dans un contexte stable, et avec SL/TP adaptés à la volatilité."

Entrées: RSI<30 + Aroon Up + Volume Oscillator > 25MA.
Sorties: Entrees mettant en oeuvre plusieurs critères sur le graphique de chaque indication.
Risk management : Stop Loss = SL = TP < Avg True Range * 1.5, avec une volatilité adaptée à la demande (ou non si c'est un marché stable).
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -100.00 | -0.639 | -92.75% | -96.19% | 0.71 | 136 | continue | ruined |
| 3 | 5 | -100.00 | -0.263 | -52.98% | -82.69% | 0.91 | 494 | continue | overtrading |
| 4 | 6 | -100.00 | -0.017 | -58.07% | -79.35% | 0.86 | 207 | stop | high_drawdown |