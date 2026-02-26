# Leaderboard Builder - session 20260225_095951_strat_gie_sur_troyusdc_1h_objective_util

Objective: Stratégie sur TROYUSDC 1h. Objective: Utilise l'AROON et le MACD pour diagnostiquer les changements de tendance sur un timeframe spécifique (1h). Incorporons une combinaison EMA + OBV pour la gestion des risques non linéaires, avec un SL / TP ajustables en fonction de la volatilité. Étudions également le comportement aleatoire à l'aide d'un filtre de session/horaire et d'une confirmation contrariante partielle sur les entrées. Entrées : AROON (20), MACD (12, 26) ; SL = TP * volatilité > X, risk management = [SL / TP] pour la gestion des risques non linéaires et SL/TP ajustables en fonction de la volatilité. Sorties : entrées avec AROON < 10 (tendance descendante) + MACD(26)<0 (tendance descendante), sorties = "Trend Down". Entrées avec AROON > 95 (tendance ascendante) + MACD(26)>0 (tendance ascendante), entrées = "Trend Up", sortie = "Range Bound" si le MACD est entre 0 et 10. Note: Ne pas utiliser de combina
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -44.53 | 0.134 | -11.49% | -48.87% | 0.95 | 65 | continue | needs_work |