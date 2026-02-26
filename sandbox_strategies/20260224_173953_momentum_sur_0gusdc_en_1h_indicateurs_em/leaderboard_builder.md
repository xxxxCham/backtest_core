# Leaderboard Builder - session 20260224_173953_momentum_sur_0gusdc_en_1h_indicateurs_em

Objective: Momentum sur 0GUSDC en 1h. Indicateurs : EMA + BOLLINGER + VOLUME_OSCILLATOR. Entrées : Croisement haussier EMA 50/200 avec compression Bollinger inférieure à 1.5σ et volume oscillateur > 0.5. Sorties : Croisement baissier EMA avec extension Bollinger supérieure à 2σ ou perte de 1.2σ sur 1h. Risk management : SL dynamique sur ATR avec TP adaptatif selon volatilité implicite. Mode risk rotation actif : seuil de volatilité 1.5x ATR pour basculer en profile risque serré. Mode inverse testé : signal inverse confirmé par Vortex + ROC > 0.1 avant entrée. Hypothèse : croisements EMA dans zone de compression Bollinger avec volume oscillateur élevé révèlent des opportunités de breakout avec forte probabilité de continuation.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -50.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 3 | 2 | -100.00 | -20.000 | -135.25% | -100.00% | 0.89 | 979 | continue | ruined |