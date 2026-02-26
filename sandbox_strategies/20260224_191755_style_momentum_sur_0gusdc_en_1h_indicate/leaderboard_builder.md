# Leaderboard Builder - session 20260224_191755_style_momentum_sur_0gusdc_en_1h_indicate

Objective: Style momentum sur 0GUSDC en 1h. Indicateurs : EMA + OBV + VOLUME_OSCILLATOR. Entrées : EMA 50>EMA 200 avec OBV en hausse et VOLUME_OSCILLATOR croissant depuis 3 périodes. Sorties : EMA 50<EMA 200 ou VOLUME_OSCILLATOR descend. Risk management : SL à 1.5x ATR, TP à 2x ATR. Mode_offbeat : utilisation combinée de OBV et VOLUME_OSCILLATOR pour détecter les mouvements de liquidité cachés. Mode_inverse : test de la logique inverse de l'indicateur OBV en contexte de volatilité élevée. Hypothèse testable : les mouvements de liquidité détectés par le volume oscillateur et OBV peuvent prédire les retournements de tendance avant que l'EMA ne change de direction.
Status: failed
Best Sharpe: 0.000
Best Continuous Score: -26.00

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4 | -50.00 | 0.000 | +0.00% | 0.00% | 0.00 | 0 | continue | no_trades |
| 2 | 1 | -85.65 | -0.702 | -33.00% | -42.94% | 0.78 | 69 | continue | wrong_direction |