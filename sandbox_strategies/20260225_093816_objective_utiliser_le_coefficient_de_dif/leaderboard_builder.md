# Leaderboard Builder - session 20260225_093816_objective_utiliser_le_coefficient_de_dif

Objective: Objective: Utiliser le coefficient de diffusion adraste (ADR) en conjonction avec les canaux d'informer sur la quantite (OBV) pour repérer des points de rupture à l'aide du filtre anti-faux-signaux (TP<0.05xATR>) et le vote majoritaire dans une direction dominantes (80% votes en partant de 2 axes). Entrées : ALGOUSDC, 30m. Conditions : ADR < -1, OBV > median(OBV), TP<0.05*ATR, [mode_risk_rotation]. Sorties: buy ou sell. Risk management : SL = TP + 0.2xATR, TP=Fibo.P[1/4], stops-loss at SL and profit target at Fibo.Retracement[38.2%].
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 2 | -100.00 | -20.000 | -227.64% | -100.00% | 0.80 | 954 | continue | ruined |
| 3 | 4 | -100.00 | -20.000 | -1032.91% | -100.00% | 0.66 | 3078 | continue | ruined |
| 4 | 5 | -100.00 | -20.000 | -833.21% | -100.00% | 0.64 | 1830 | continue | ruined |