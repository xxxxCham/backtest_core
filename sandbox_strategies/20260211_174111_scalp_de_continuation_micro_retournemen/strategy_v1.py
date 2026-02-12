class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalp_de_continuation_micro_retournement")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"ema_period": 21, "rsi_overbought": 70, "rsi_oversold": 30, "stop_atr_mult": 1.0, "tp_atr_mult": 2.0}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # Define ParameterSpec for each tunable parameter
            # Example: "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # IMPLEMENT: Entry LONG conditions → signals = 1.0
        # IMPLEMENT: Entry SHORT conditions → signals = -1.0
        # IMPLEMENT: Exit / flat conditions → signals = 0.0

        return signals