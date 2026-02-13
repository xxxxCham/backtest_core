class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")
        
    @property
    def required_indicators(self) -> List[str]:
        return ['KELTNER', 'CCI', 'ATR']
    
    @property
    def default_params(self) -> Dict[str, Any]:
        return {}
    
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, np.ndarray], params: Dict[str, Any]) -> pd.Series:
        
        # Keltner values
        keltner = indicators['KELTNER']
        bollinger = indicators['bollinger']['upper']
        adx_adx = indicators['ADX']['adx']
        supertrend_direction = indicators['SUPERTREND']['direction']
        
        # CCI values
        cci = indicators['CCI']
        
        # ATR values
        atr = indicators['ATR']
    
        signals = pd.Series(0.0, index=df.index)
        
        for i in range(1, len(df)):
            if keltner[i] < keltner[i-1]:  # Keltner is extreme
                if cci[i] > cci[i+1] and adx_adx[i] > adx_adx[i+1] and supertrend_direction[i] == 'up':  # CCI shows a reversal then enter long.
                    signals[i] = 1.0
                elif cci[i] < cci[i-1]:   # price moves back towards the mean or above the keltner channel's upper band.
                    signals[i] = -1.0
            else:
                if adx_adx[i] > adx_adx[i+1] and supertrend_direction[i] == 'down':  # Keltner is extreme then enter short.
                    signals[i] = -1.0
                
                elif atr[i-params['warmup']] / params['atr'].mean() > 2 or atr[i-params['warmup']] / params['atr'].mean() < 0.5:  # ATR crosses above/below 2x/5x its average over the last 14 periods
                    signals[i] = self.exit_long(signals, i) if signals[i] == 1.0 else -self.exit_short(signals, i)  
                
        return signals
    
    def exit_long(self, signals: pd.Series, idx: int):
        close_price = df['close'][idx]
        
        if close_price > df['close'].iloc[params['rsi_overbought']]:  # price moves back towards the mean or above the keltner channel's upper band and in an uptrending market.
            signals[idx] = -1.0
            
    def exit_short(self, signals: pd.Series, idx: int):
        close_price = df['close'][idx]
        
        if close_price < df['close'].iloc[params['rsi_oversold']]:  # price moves back towards the mean or above the keltner channel's upper band and in a downtrending market.
            signals[idx] = -1.0  
            
    return signals