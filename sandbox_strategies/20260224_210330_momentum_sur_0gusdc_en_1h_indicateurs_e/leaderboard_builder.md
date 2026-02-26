# Leaderboard Builder - session 20260224_210330_momentum_sur_0gusdc_en_1h_indicateurs_e

Objective: [Momentum] sur 0GUSDC en 1h. Indicateurs: EMA + OBV + STOCHASTIC. Entrées : adx > 35, aroon_up > 80, atr_1h < 2%, cci > -200, fear_greed <= 7, macd > signal_hist, mfi < 20, momentum > 90. Sorties: vwap < 0.5 * sma(vwap, 4), volume_oscillator < 1/3 * ema(volume_oscillator, 8). Risk management : SL = 6%, TP = 7%.

L'objectif est de tester une combinaison inhabituelle d'indicateurs qui se sont avérés performants sur le marché des crypto-monnaies. On combine l'EMA pour les courbes moyennes, OBV et STOCHASTIC pour la volatilité, ainsi qu'un ensemble de filtres indiquant une tendance positive ou negative. On place ensuite un stop loss à 6% du coût et un target a l'aide d'une règle t+p fixe qui permet de profiter des variations rapides sans s'exposer trop longtemps aux changes majeures sur le march
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 51.45 | 0.485 | +48.09% | -40.15% | 1.06 | 336 | continue | overtrading |
| 2 | 2 | -100.00 | -20.000 | -173.75% | -100.00% | 0.64 | 92 | continue | ruined |
| 3 | 6 | -100.00 | -20.000 | -65.16% | -100.00% | 0.89 | 181 | continue | ruined |