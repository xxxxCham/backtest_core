class BuilderGeneratedStrategy(StrategyBase):
    @property
    def required_indicators(self) -> List[str]:
        return ["keltner", "cci"]
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"cci_bullish_divergence_threshold": 25, "keltner_overbought_threshold": 70, "warmup_period": 40}
    
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        # fill each tunable parameter with valid values
        pass
        
    def generate_signals(
        self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any] 
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        
        # Implement explicit LONG / SHORT / FLAT logic
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
            
        return signals