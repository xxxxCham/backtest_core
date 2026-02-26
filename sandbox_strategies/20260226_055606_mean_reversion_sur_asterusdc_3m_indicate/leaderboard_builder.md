# Leaderboard Builder - session 20260226_055606_mean_reversion_sur_asterusdc_3m_indicate

Objective: Mean-reversion sur ASTERUSDC 3m. Indicateurs : KELTNER + AROON + ATR. Entrées : Croisement baissier des bandes de Keltner Channels confirmé par un Aroon Down > 70% et une divergence haussière sur l'Onchain Smoothing. Sorties : Clôture au-dessus de la bande de Keltner Channels centrale, ou retour à la moyenne mobile exponentielle sur un timeframe plus court, avec une confirmation inverse partielle du RSI (RSI > 50 alors que l'Aroon Down est toujours > 60%). Risk management : Stop Loss initialement fixé à 1.5x l'ATR, Take Profit dynamique basé sur les niveaux de Fibonacci retracement du mouvement précédent, avec une rotation du profil de risque : SL serré en période de faible volatilité, SL large en période de forte volatilité.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | stop | no_trades |
| 2 | 2 | -100.00 | -20.000 | -1457.68% | -100.00% | 0.31 | 5360 | continue | ruined |
| 3 | 3 | -100.00 | -20.000 | -505.90% | -100.00% | 0.52 | 1956 | continue | ruined |
| 4 | 5 | -100.00 | -20.000 | -504.08% | -100.00% | 0.52 | 1938 | continue | ruined |