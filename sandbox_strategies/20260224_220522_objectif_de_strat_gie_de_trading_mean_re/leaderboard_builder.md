# Leaderboard Builder - session 20260224_220522_objectif_de_strat_gie_de_trading_mean_re

Objective: Objectif de stratégie de trading :
[Mean-reversion] sur [0GUSDC] dans les 1h.[ind1] + [ind3].[ind1] = Bollinger, [ind2] = EMA,[ind3] = ROC, Entrées : [bollinger_upper], [bollinger_lower], [ema_period],[roc_length]. Sorties : [signal]. Risk management : [SL=0.5*atr, TP=(BOP + 1.5*EMA)].[ind2] = Aroon, Entrées : [arooon_upper], [arooon_lower], [ema_period],[roc_length]. Sorties : [signal].

Créativement, cette stratégie explore des combinaisons inhabituelles en utilisant deux filtres de Bollinger et ROC pour l'analyse sur une période courte (1h), ainsi qu'un filtre plus long basé sur Aroon. Les entrées prévoient le point supérieur et inférieur des bandes Bollinger, la durée de l'EMA pour ROC et celle de l'Aroon pour un filtre plus long. La sortie sera composée d'une stratégie de mean-reversion avec des seuils bas et hauts bas
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 2 | -100.00 | -0.639 | -92.75% | -96.19% | 0.71 | 136 | continue | ruined |
| 3 | 4 | -100.00 | -0.263 | -52.98% | -82.69% | 0.91 | 494 | continue | overtrading |