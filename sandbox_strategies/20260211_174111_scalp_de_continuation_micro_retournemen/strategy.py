class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        if (df['close'] > indicators['ema']) & (indicators['rsi'] < 30):  # Fill in with actual logic
            signals[df['close'] > indicators['ema']] = 1
        elif (df['close'] < indicators['ema']) & (indicators['rsi'] > 70):
            signals[df['close'] < indicators['ema']] = -1

        return signals