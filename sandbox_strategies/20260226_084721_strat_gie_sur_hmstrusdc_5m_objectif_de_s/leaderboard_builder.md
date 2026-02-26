# Leaderboard Builder - session 20260226_084721_strat_gie_sur_hmstrusdc_5m_objectif_de_s

Objective: Stratégie sur HMSTRUSDC 5m. Objectif de stratégie de trading: Sur le marché (token) et le timeframe spécifié automatiquement : utiliser l'indicateur ROC avec un threshold = 5% pour détecter des situations d'oscillation fortement positive, et EMA avec un period = 20 pour suivre les tendances. Incorporons ensuite la famille robuste de breakouts (utilisant le signal de Aroon) pour capturer les régions changeantes ou les zones où l'amplitude augmente fortement. Pour ajouter une dimension d'oscillation intermédiaire, nous appliquerons le filtre anti-faux-signaux basé sur la volatilité implicite/realisee pour éviter les situations de false breakout. Utiliser ensuite ROC avec threshold = 5% pour repérer les oscillations fortement positives. Comme indication additionnelle, nous allons intégrer le signal MACD qui est connu pour être un indicateur puissant des transitions majeures entre marchés en croissance et déclin. Enfin, appliquer l'indicateur ATR avec un period = 14 pour mesurer la volatilité globale du marché, qui peut nous aider à établir les stop loss (SL)
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -883.83% | -100.00% | 0.47 | 3051 | accept | ruined |