# Leaderboard Builder - session 20260226_085026_objectif_utiliser_les_indicateurs_ema_bo

Objective: Objectif : "Utiliser les indicateurs EMA + BOLLINGER + MA pour un trading à Court Terme sur JUPUSDC 5m (15m) dans une combinaison originale avec filtre horaire de liquidite et comportements aleatoires imposés." Entrées: - Indicadores EMA, BOLLINGER, SMA. - Filtro horario de liquidez a cada hora entre as 9 e das 17 horas (horarios de negocio). - Condicoes: Bollinger High + SMA(20) <= Volumen > SMA(60), EMA(3) < EMA(14). Sorties : Venta em caso de um crossover entre a Bollinger Band e o volume. TP = Preco+SL/Preco-TP, com SL sendo 5% sobre o mais alto ou baixo do candlestick ao dia seguinte à transacao. Risk management: SL=0,02 et TP = price +/- 0,01
Status: max_iterations
Best Sharpe: -inf
Best Continuous Score: -inf

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | -100.00 | -20.000 | -1001.51% | -100.00% | 0.61 | 4249 | continue | ruined |