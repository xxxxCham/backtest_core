# Code théorique pour bollinger_atr parameter_specs
# Basé sur les standards de l'analyse technique

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        """Spécifications basées sur la théorie de l'analyse technique.

        🎓 RANGES THÉORIQUES optimisés :
        - Basé sur les standards de John Bollinger et Welles Wilder
        - Évite les valeurs aberrantes des backtests (entry_z<0.5, k_sl négatif)
        - Réduit l'espace de recherche à ~6,124,608 combinaisons viables
        - Focus sur les plages utilisées par les traders professionnels

        ⚠️ ATTENTION : Les résultats backtests montrent 95.1% d'échecs.
        Cette stratégie nécessite peut-être une révision fondamentale de sa logique.
        """
        return {{
            "bb_period": ParameterSpec(
                name="bb_period",
                min_val=15, max_val=35, default=20,  # 🎓 Théorique: John Bollinger recommande 20 périodes comme standard
                param_type="int",
                description="Période des Bandes de Bollinger"
            ),
            "bb_std": ParameterSpec(
                name="bb_std",
                min_val=1.8, max_val=2.5, default=2.0,  # 🎓 Théorique: 2.0 capture ~95% des mouvements, 1.8-2.5 couvre différents régimes
                param_type="float",
                description="Écarts-types pour les bandes"
            ),
            "entry_z": ParameterSpec(
                name="entry_z",
                min_val=1.5, max_val=2.2, default=2.0,  # 🎓 Théorique: 1.5-2.2 permet variations autour de la band standard
                param_type="float",
                description="Seuil z-score pour entree"
            ),
            "atr_period": ParameterSpec(
                name="atr_period",
                min_val=10, max_val=21, default=14,  # 🎓 Théorique: 14 périodes recommandé par Wilder, 10-21 couvre court/moyen terme
                param_type="int",
                description="Période de l'ATR"
            ),
            "atr_percentile": ParameterSpec(
                name="atr_percentile",
                min_val=20, max_val=50, default=30,  # 🎓 Théorique: 20-50 filtre les marchés trop calmes/agités
                param_type="int",
                description="Percentile volatilite minimum (ATR)"
            ),
            "k_sl": ParameterSpec(
                name="k_sl",
                min_val=1.2, max_val=2.5, default=1.5,  # 🎓 Théorique: 1.2-2.5 ATR couvre différents styles de gestion du risque
                param_type="float",
                description="Multiplicateur ATR pour stop-loss"
            ),
            "leverage": ParameterSpec(
                name="leverage",
                min_val=1, max_val=10, default=1,
                param_type="int",
                description="Levier de trading (non optimisé)",
                optimize=False,
            ),
        }