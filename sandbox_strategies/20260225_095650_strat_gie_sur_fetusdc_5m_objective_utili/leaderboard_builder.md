# Leaderboard Builder - session 20260225_095650_strat_gie_sur_fetusdc_5m_objective_utili

Objective: Stratégie sur FETUSDC 5m. Objective: Utiliser la combination de Bollinger Bands et Donchian Channels avec l'EMA sur une hypothese d'un changement dans le cours pour un token DOGEUSDT, timeframe. Les entrées seront le volume_oscillator, Aroon Up et donc les sorties auront des SL/TP en fonction de la volatilité du marché à chaque fois que l'EMA change d'orientation sur Bollinger Bands ou Donchian Channels. Entrées : 1. EMA(3) > EMA(20), 2. BBANDS(BB_n=5, ndevs=2).3. DC((HH-LL)/2:DDT, timeframe=timeframe) >= x. Sorties : 1. SL = HH - (x/7*volume_oscillator) ou TP = LL + volume_oscillator avec la différence entre le plus haut et le plus bas historique de l'oscillateur.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -965.00% | -100.00% | 0.61 | 3778 | continue | ruined |
| 2 | 6 | -100.00 | -20.000 | -5560.81% | -100.00% | 0.43 | 15701 | stop | ruined |