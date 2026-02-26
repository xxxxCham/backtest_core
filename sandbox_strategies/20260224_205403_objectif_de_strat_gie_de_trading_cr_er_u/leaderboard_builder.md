# Leaderboard Builder - session 20260224_205403_objectif_de_strat_gie_de_trading_cr_er_u

Objective: Objectif de stratégie de trading: Créer une stratégie sur le marché 0GUSDC dans l'intervalle d'1h en utilisant les indicateurs suivants : ADR, DMI, EMA + OBV et ROC. Les entrées seront basées sur la volatility horaire de liquidité (qrtz_timezones) avec un stop loss à 3% sous le niveau d'achat (SL1=0.97*pv2) et un take profit a l'échelle d'un jour (tp = dt[6]). Le risk management sera basé sur la zone de confiance minimale de volatilité (min_volatility).

Pour répondre à cette demande, une stratégie de trading doit être créée pour le marché 0GUSDC dans l'intervalle d'1h. Pour ce faire, plusieurs indicateurs sont proposés tels que ADR, DMI et EMA + OBV ainsi qu'ROC. La volatility horaire de liquidité est utilisée pour les entrées des stratégies en fonction du temps quartier (qrtz_timezones). Un stop loss à 3% sous le niveau d'achat et un take profit sur
Status: failed
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 5.15 | 0.844 | +17.44% | -67.19% | 1.06 | 112 | continue | high_drawdown |
| 2 | 3 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 3 | 2 | -68.42 | -0.320 | -25.20% | -43.53% | 0.86 | 68 | continue | wrong_direction |
| 4 | 4 | -100.00 | -20.000 | -104.11% | -100.00% | 0.78 | 260 | continue | ruined |