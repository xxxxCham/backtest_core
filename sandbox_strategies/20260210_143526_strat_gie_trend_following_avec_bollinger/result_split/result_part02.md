      "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_periods": [21, 50],
            "rsi_period": 14,
            "stop_loss_mult": 2,
            "take_profit_mult": 1.5
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(type=float, min=10, max=30),
            "ema_periods": ParameterSpec(type=list, default=[21, 50]),
            "rsi_period": ParameterSpec(type=int, min=10, max=30),
            "stop_loss_mult": ParameterSpec(type=float, min=1, max=3),
            "take_profit_mult": ParameterSpec(type=float, min=1, max=2)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get precomputed indicators
        ema_indicator = indicators["ema"]
        rsi_val = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]

        # Extract EMA values for periods 21 and 50
        ema_21 = np.nan_to_num(ema_indicator[params.get("ema_periods")[0]])
        ema_50 = np.nan_to_num(ema_indicator[params.get("ema_periods")[1]])

        # Bollinger Bands values
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])

        # Entry conditions
        long_entry = (
            (ema_21 > ema_50) &
            (df["close"].values > lower_band) &
            (rsi_val < params.get("long_rsi_threshold", 35))
        )

        short_entry = (
            (ema_21 < ema_50) &
            (df["close"].values < upper_band) &
            (rsi_val > params.get("short_rsi_threshold", 65))
        )

        # Exit conditions
        exit_long = df["close"].values >= upper_band
        exit_short = df["close"].values <= lower_band

        for i in range(n):
            if i < max(params.get("ema_periods")):
                signals[i] = 0.0
                continue

            current_close = df["close"].values[i]

            # Check entry conditions
            if long_entry[i]:
                signals[i] = 1.0
            elif short_entry[i]:
                signals[i] = -1.0

            # Check exit conditions
            if signals[i-1] == 1.0 and (exit_long[i] or current_close <= lower_band[i]):
                signals[i] = 0.0
            elif signals[i-1] == -1.0 and (exit_short[i] or current_close >= upper_band[i]):
                signals[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_143609_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_143609_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ContinuationScalpStrategy
    # Objective: Capture short-term continuations or micro-reversals using EMA, RSI, and Bollinger Bands

    def __init__(self):
        super().__init__(name="ContinuationScalpStrategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "minimum_risk_reward_ratio": 1.5,
            "risk_percentage_per_trade": 1,
            "stop_loss_multiplier": 1,
            "ema_period": 21,
            "rsi_overbought": 70,
            "rsi_oversold": 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "minimum_risk_reward_ratio": ParameterSpec(
                name="Minimum Risk/Reward Ratio",
                current_value=1.5,
                min_value=1.0,
                max_value=3.0,
                type=float,
                tunable=True
            ),
            "risk_percentage_per_trade": ParameterSpec(
                name="Risk Percentage Per Trade",
                current_value=1,
                min_value=0.5,
                max_value=2,
                type=float,
                tunable=True
            ),
            "stop_loss_multiplier": ParameterSpec(
                name="Stop Loss Multiplier",
                current_value=1,
                min_value=0.5,
                max_value=2,
                type=float,
                tunable=True
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any]
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required data
        close = np.array(df["close"], dtype=np.float64)
        high = np.array(df["high"], dtype=np.float64)
        low = np.array(df["low"], dtype=np.float64)

        # Access indicators
        ema21 = np.nan_to_num(indicators["ema"][params.get("ema_period", 21)])
        rsi = np.nan_to_num(indicators["rsi"])

        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])

        # Initialize entry conditions
        for i in range(1, n):
            if signals[i-1] == 0:
                # Look for new entries
                # LONG condition
                if (
                    close[i] > ema21[i] and
                    rsi[i] > params.get("rsi_oversold", 30) and
                    (close[i-1] < upper_bb[i-1] or low[i] < upper_bb[i])
                ):
                    signals[i] = 1.0

                # SHORT condition
                elif (
                    close[i] < ema21[i] and
                    rsi[i] < params.get("rsi_overbought", 70) and
                    (close[i-1] > lower_bb[i-1] or high[i] > lower_bb[i])
                ):
                    signals[i] = -1.0

            else:
                # Check exit conditions
                current_position = signals[i-1]

                if current_position == 1.0:
                    # Exit LONG
                    if close[i] >= upper_bb[i]:
                        signals[i] = 0.0
                    elif rsi[i] < rsi[i-1] and high[i] > high[i-1]:
                        signals[i] = 0.0

                elif current_position == -1.0:
                    # Exit SHORT
                    if close[i] <= lower_bb[i]:
                        signals[i] = 0.0
                    elif rsi[i] > rsi[i-1] and low[i] < low[i-1]:
                        signals[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_143609_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_143609_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ScalpContinuationBands
    # Objective: Scalp de continuation/micro-retournement on liquid crypto (BTCUSDC)
    # Timeframe: 30m

    def __init__(self):
        super().__init__(name="ScalpContinuationBands")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "ema", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_period_short": 21,
            "ema_period_long": 50,
            "rsi_period": 14,
            "risk_percentage": 1.5,
            "stop_loss_type": "EMA",
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(int, default=20, min=10, max=30),
            "ema_period_short": ParameterSpec(int, default=21, min=10, max=50),
            "ema_period_long": ParameterSpec(int, default=50, min=30, max=100),
            "rsi_period": ParameterSpec(int, default=14, min=7, max=28),
            "risk_percentage": ParameterSpec(float, default=1.5, min=1.0, max=2.0),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get required data
        close = df["close"].values
        ema_short = np.nan_to_num(indicators["ema"])
        ema_long = np.nan_to_num(indicators["ema"])  # EMA50
        rsi = np.nan_to_num(indicators["rsi"])

        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])

        # Calculate EMA21 and EMA50 crossover conditions
        ema_short_condition = close > ema_short
        ema_long_condition = close > ema_long

        # RSI conditions for trend continuation
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70

        # Bollinger band rejection conditions
        lower_bb_reject = (close[-2] < lower_bb[-2]) & (close[-1] > close[-2])
        upper_bb_reject = (close[-2] > upper_bb[-2]) & (close[-1] < close[-2])

        # Generate LONG entries
        long_entry = (
            ema_short_condition &
            ~ema_long_condition &
            rsi_oversold &
            lower_bb_reject
        )

        # Generate SHORT entries
        short_entry = (
            ~ema_short_condition &
            ema_long_condition &
            rsi_overbought &
            upper_bb_reject
        )

        # Set signals for valid entries (starting from index 50 to ensure data)
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Exit conditions: opposite band or divergence
        exit_long = close > upper_bb
        exit_short = close < lower_bb

        # Apply exits by setting signals back to 0
        signals[exit_long] = 0.0
        signals[exit_short] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_143609_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_143609_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    # Auto-generated strategy: ScalpContinuationBandsV2
    # Objective: Scalp de continuation/micro-retournement using EMA, RSI, and Bollinger Bands

    def __init__(self):
        super().__init__(name="ScalpContinuationBandsV2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "bollinger_std_dev": 2,
            "ema_periods": [21, 50],
            "rsi_period": 14,
            "stop_loss_mult": 2,
            "take_profit_mult": 1.5
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(type=float, min=10, max=30),
            "ema_periods": ParameterSpec(type=list, default=[21, 50]),
            "rsi_period": ParameterSpec(type=int, min=10, max=30),
            "stop_loss_mult": ParameterSpec(type=float, min=1, max=3),
            "take_profit_mult": ParameterSpec(type=float, min=1, max=2)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get precomputed indicators
        ema_indicator = indicators["ema"]
        rsi_val = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]

        # Extract EMA values for periods 21 and 50
        ema_21 = np.nan_to_num(ema_indicator[params.get("ema_periods")[0]])
        ema_50 = np.nan_to_num(ema_indicator[params.get("ema_periods")[1]])

        # Bollinger Bands values
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])

        # Entry conditions
        long_entry = (
            (ema_21 > ema_50) &
            (df["close"].values > lower_band) &
            (rsi_val < params.get("long_rsi_threshold", 35))
        )

        short_entry = (
            (ema_21 < ema_50) &
            (df["close"].values < upper_band) &
            (rsi_val > params.get("short_rsi_threshold", 65))
        )

        # Exit conditions
        exit_long = df["close"].values >= upper_band
        exit_short = df["close"].values <= lower_band

        for i in range(n):
            if i < max(params.get("ema_periods")):
                signals[i] = 0.0
                continue

            current_close = df["close"].values[i]

            # Check entry conditions
            if long_entry[i]:
                signals[i] = 1.0
            elif short_entry[i]:
                signals[i] = -1.0

            # Check exit conditions
            if signals[i-1] == 1.0 and (exit_long[i] or current_close <= lower_band[i]):
                signals[i] = 0.0
            elif signals[i-1] == -1.0 and (exit_short[i] or current_close >= upper_band[i]):
                signals[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_152905_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_152905_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Dict

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="MicroRetournementBTC30m")

    @property
    def required_indicators(self) -> list:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> dict:
        return {
            "atr_period": 14,
            "bb_period": 20,
            "bb_stddev": 2,
            "ema_period": 21,
            "rsi_period": 14,
            "risk_percent": 1
        }

    @property
    def parameter_specs(self) -> dict:
        return {
            "atr_period": ParameterSpec(name="atr_period", default=14, description="ATR period"),
            "bb_period": ParameterSpec(name="bb_period", default=20, description="Bollinger period"),
            "bb_stddev": ParameterSpec(name="bb_stddev", default=2, description="Bollinger std dev"),
            "ema_period": ParameterSpec(name="ema_period", default=21, description="EMA period"),
            "rsi_period": ParameterSpec(name="rsi_period", default=14, description="RSI period"),
            "risk_percent": ParameterSpec(name="risk_percent", default=1, description="Risk percent per trade")
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = np.zeros(n, dtype=np.float64)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        atr_val = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])

        for i in range(1, n):
            # Long: pullback to EMA21, RSI crossing up through 50, rejection of lower band
            if (rsi_val[i] > 50 and rsi_val[i-1] <= 50) and \
               close.iloc[i] > ema_val[i] and \
               close.iloc[i] > lower[i] and \
               close.iloc[i-1] <= lower[i-1]:
                signals[i] = 1.0
            # Short: spike to upper band, RSI crossing down from overbought, reversal below upper band but above EMA21
            elif (rsi_val[i] < 70 and rsi_val[i-1] >= 70) and \
                 high.iloc[i] >= upper[i] and \
                 close.iloc[i] < upper[i] and close.iloc[i] > ema_val[i]:
                signals[i] = -1.0

        return pd.Series(signals, index=df.index)
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_152905_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_152905_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Dict

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="MicroRetournementBTC30m")

    @property
    def required_indicators(self) -> list:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> dict:
        return {
            "atr_period": 14,
            "bb_period": 20,
            "bb_stddev": 2,
            "ema_period": 21,
            "rsi_period": 14,
            "risk_percent": 1
        }

    @property
    def parameter_specs(self) -> dict:
        return {
            "atr_period": ParameterSpec(name="atr_period", default=14, description="ATR period"),
            "bb_period": ParameterSpec(name="bb_period", default=20, description="Bollinger period"),
            "bb_stddev": ParameterSpec(name="bb_stddev", default=2, description="Bollinger std dev"),
            "ema_period": ParameterSpec(name="ema_period", default=21, description="EMA period"),
            "rsi_period": ParameterSpec(name="rsi_period", default=14, description="RSI period"),
            "risk_percent": ParameterSpec(name="risk_percent", default=1, description="Risk percent per trade")
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        n = len(df)
        signals = np.zeros(n, dtype=np.float64)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        atr_val = np.nan_to_num(indicators["atr"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])

        for i in range(1, n):
            # Long: pullback to EMA21, RSI crossing up through 50, rejection of lower band
            if (rsi_val[i] > 50 and rsi_val[i-1] <= 50) and \
               close.iloc[i] > ema_val[i] and \
               close.iloc[i] > lower[i] and \
               close.iloc[i-1] <= lower[i-1]:
                signals[i] = 1.0
            # Short: spike to upper band, RSI crossing down from overbought, reversal below upper band but above EMA21
            elif (rsi_val[i] < 70 and rsi_val[i-1] >= 70) and \
                 high.iloc[i] >= upper[i] and \
                 close.iloc[i] < upper[i] and close.iloc[i] > ema_val[i]:
                signals[i] = -1.0

        return pd.Series(signals, index=df.index)
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_174111_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_174111_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
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
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_174111_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_174111_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
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
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_174111_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_174111_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        ema_short = indicators['ema'][1]
        ema_long = indicators['ema'][2]
        rsi = indicators['rsi'][1]

        conditions = [
            (df.close > ema_short) & (df.close < df.local_max),
            (df.close > ema_short) & (df.close < ema_long),
            (rsi > 80)
        ]
        action = ['buy', 'buy', 'sell']

        for i, cond in enumerate(conditions):
            signals.loc[(cond), :] = action[i]

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_174111_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_174111_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
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
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_174335_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_174335_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalp_crypto_ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators and handle NaNs
        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])

        close = df["close"].values

        # Entry conditions
        long_entry = (close > ema_val) & (rsi_val < params["rsi_overbought"]) & (
            (close > upper) | (close < lower)
        )
        short_entry = (close < ema_val) & (rsi_val > params["rsi_oversold"]) & (
            (close > upper) | (close < lower)
        )

        # Exit conditions
        is_long = signals == 1.0
        is_short = signals == -1.0

        long_exit = (close < lower) | (rsi_val > params["rsi_overbought"])
        short_exit = (close > upper) | (rsi_val < params["rsi_oversold"])

        # Update signals
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        signals[is_long & long_exit] = 0.0
        signals[is_short & short_exit] = 0.0

        # Warmup period
        warmup = 50
        signals.iloc[:warmup] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_174335_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_174335_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Get indicators dataframes
        ema_df = indicators['ema']
        rsi_df = indicators['rsi']
        bollinger_df = indicators['bollinger']

        # Ensure all dataframes are aligned with the main df
        ema_21 = ema_df['ema_21'].align(df['close'], join='left')[0]
        rsi = rsi_df['rsi_14'].align(df['close'], join='left')[0]
        lower_band = bollinger_df['lower'].align(df['close'], join='left')[0]
        upper_band = bollinger_df['upper'].align(df['close'], join='left')[0]

        # Calculate previous values for crossover detection
        prev_rsi = rsi.shift(1)

        # LONG signals
        long_conditions = (
            df['close'] > ema_21,
            (rsi > 50) & (prev_rsi <= 50),
            df['close'] > lower_band,
            df['close'].shift(1) < df['close'],
            df['close'] > lower_band.shift(1)
        )

        # SHORT signals
        short_conditions = (
            df['close'] < ema_21,
            (rsi < 50) & (prev_rsi >= 50),
            df['close'] < upper_band,
            df['close'].shift(1) > df['close'],
            df['close'] < upper_band.shift(1)
        )

        # Generate signals
        long_signals = pd.Series(0.0, index=df.index)
        short_signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            if all(long_conditions[j][i] for j in range(len(long_conditions))):
                long_signals.iloc[i] = 1.0
            if all(short_conditions[j][i] for j in range(len(short_conditions))):
                short_signals.iloc[i] = -1.0

        signals = long_signals + short_signals

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_174335_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_174335_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="scalp_crypto_ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        n = len(df)
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators and handle NaNs
        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        lower = np.nan_to_num(bb["lower"])

        close = df["close"].values

        # Entry conditions
        long_entry = (close > ema_val) & (rsi_val < params["rsi_overbought"]) & (
            (close > upper) | (close < lower)
        )
        short_entry = (close < ema_val) & (rsi_val > params["rsi_oversold"]) & (
            (close > upper) | (close < lower)
        )

        # Exit conditions
        is_long = signals == 1.0
        is_short = signals == -1.0

        long_exit = (close < lower) | (rsi_val > params["rsi_overbought"])
        short_exit = (close > upper) | (rsi_val < params["rsi_oversold"])

        # Update signals
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        signals[is_long & long_exit] = 0.0
        signals[is_short & short_exit] = 0.0

        # Warmup period
        warmup = 50
        signals.iloc[:warmup] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_175901_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_175901_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'rsi_overbought': 67,
         'rsi_oversold': 33,
         'rsi_period': 12,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.6,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
            "rsi_oversold": ParameterSpec(min_val=10, max_val=45, default=35, param_type="int"),
            "rsi_overbought": ParameterSpec(min_val=55, max_val=90, default=65, param_type="int"),
            "stop_atr_mult": ParameterSpec(min_val=0.5, max_val=5.0, default=1.5, param_type="float"),
            "tp_atr_mult": ParameterSpec(min_val=0.8, max_val=8.0, default=2.4, param_type="float"),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        mean_revert_exit = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[mean_revert_exit] = 0.0

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_175901_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_175901_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 2.4,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
            "rsi_oversold": ParameterSpec(min_val=10, max_val=45, default=35, param_type="int"),
            "rsi_overbought": ParameterSpec(min_val=55, max_val=90, default=65, param_type="int"),
            "stop_atr_mult": ParameterSpec(min_val=0.5, max_val=5.0, default=1.5, param_type="float"),
            "tp_atr_mult": ParameterSpec(min_val=0.8, max_val=8.0, default=2.4, param_type="float"),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        mean_revert_exit = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[mean_revert_exit] = 0.0

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_175901_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_175901_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'rsi_overbought': 67,
         'rsi_oversold': 33,
         'rsi_period': 12,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.6,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int"),
            "rsi_oversold": ParameterSpec(min_val=10, max_val=45, default=35, param_type="int"),
            "rsi_overbought": ParameterSpec(min_val=55, max_val=90, default=65, param_type="int"),
            "stop_atr_mult": ParameterSpec(min_val=0.5, max_val=5.0, default=1.5, param_type="float"),
            "tp_atr_mult": ParameterSpec(min_val=0.8, max_val=8.0, default=2.4, param_type="float"),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        mean_revert_exit = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))

        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[mean_revert_exit] = 0.0

        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_180646_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_180646_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'rsi_overbought': 67,
         'rsi_oversold': 33,
         'rsi_period': 12,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.6,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int")}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        exit_cond = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[exit_cond] = 0.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_180646_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_180646_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period": 14, "rsi_oversold": 35, "rsi_overbought": 65, "stop_atr_mult": 1.5, "tp_atr_mult": 2.4, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int")}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        exit_cond = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[exit_cond] = 0.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_180646_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_180646_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="builder_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'rsi_overbought': 67,
         'rsi_oversold': 33,
         'rsi_period': 12,
         'stop_atr_mult': 1.3,
         'tp_atr_mult': 2.6,
         'warmup': 50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5, max_val=50, default=14, param_type="int")}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        close = df["close"].values
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        ema = np.nan_to_num(indicators["ema"])
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        long_cond = (close > ema) & (rsi < params.get("rsi_oversold", 35)) & (close < lower)
        short_cond = (close < ema) & (rsi > params.get("rsi_overbought", 65)) & (close > upper)
        exit_cond = ((close > middle) & (rsi > 50)) | ((close < middle) & (rsi < 50))
        signals[long_cond] = 1.0
        signals[short_cond] = -1.0
        signals[exit_cond] = 0.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_180746_mini\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_180746_mini\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mini")
    @property
    def required_indicators(self) -> List[str]:
        return ["rsi","bollinger","atr"]
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period":14,"rsi_oversold":30,"rsi_overbought":70,"stop_atr_mult":1.5,"tp_atr_mult":3.0,"warmup":50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5,max_val=50,default=14,param_type="int")}
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        close = df["close"].values
        signals[(rsi < params.get("rsi_oversold", 30)) & (close < lower)] = 1.0
        signals[(rsi > params.get("rsi_overbought", 70)) & (close > upper)] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_180746_mini\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_180746_mini\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from utils.parameters import ParameterSpec
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="mini")
    @property
    def required_indicators(self) -> List[str]:
        return ["rsi","bollinger","atr"]
    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_period":14,"rsi_oversold":30,"rsi_overbought":70,"stop_atr_mult":1.5,"tp_atr_mult":3.0,"warmup":50}
    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {"rsi_period": ParameterSpec(min_val=5,max_val=50,default=14,param_type="int")}
    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        close = df["close"].values
        signals[(rsi < params.get("rsi_oversold", 30)) & (close < lower)] = 1.0
        signals[(rsi > params.get("rsi_overbought", 70)) & (close > upper)] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_181105_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_181105_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                type="int",
                min=50,
                max=90,
                default=70,
                description="RSI overbought level"
            ),
            "rsi_oversold": ParameterSpec(
                type="int",
                min=10,
                max=50,
                default=30,
                description="RSI oversold level"
            ),
            "rsi_period": ParameterSpec(
                type="int",
                min=2,
                max=50,
                default=14,
                description="RSI period"
            ),
            "stop_atr_mult": ParameterSpec(
                type="float",
                min=1.0,
                max=3.0,
                default=1.5,
                description="Stop loss multiplier based on ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type="float",
                min=2.0,
                max=5.0,
                default=3.0,
                description="Take profit multiplier based on ATR"
            ),
            "warmup": ParameterSpec(
                type="int",
                min=20,
                max=100,
                default=50,
                description="Warmup period to ignore initial signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators with NaN protection
        bollinger = np.nan_to_num(indicators["bollinger"])
        rsi = np.nan_to_num(indicators["rsi"])
        ema = np.nan_to_num(indicators["ema"])

        for i in range(warmup, len(df)):
            # Current price conditions
            upper_band = bollinger[i, 0]
            lower_band = bollinger[i, 1]
            close = df.iloc[i].close

            # EMA condition
            ema_21 = ema[i, 1]  # Assuming ema[period=21] is second element

            # RSI conditions
            rsi_ob = params["rsi_overbought"]
            rsi_os = params["rsi_oversold"]

            # Check for long entry
            if (close > upper_band) and (rsi[i] < rsi_os) and (close > ema_21):
                signals.iloc[i] = 1.0  # LONG

            # Check for short entry
            elif (close < lower_band) and (rsi[i] > rsi_ob) and (close < ema_21):
                signals.iloc[i] = -1.0  # SHORT

            # Exit conditions
            elif (close < upper_band and signals.iloc[i-1] == 1.0) or \
                 (close > lower_band and signals.iloc[i-1] == -1.0):
                signals.iloc[i] = 0.0  # FLAT

            # Stop loss conditions based on Bollinger reversal
            elif ((signals.iloc[i-1] == 1.0 and close < lower_band) or
                  (signals.iloc[i-1] == -1.0 and close > upper_band)):
                signals.iloc[i] = 0.0  # FLAT

            # RSI divergence exit
            elif (rsi[i] < rsi_ob and signals.iloc[i-1] == -1.0) or \
                 (rsi[i] > rsi_os and signals.iloc[i-1] == 1.0):
                signals.iloc[i] = 0.0  # FLAT

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_181105_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_181105_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'ema', 'rsi']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Get indicator data
        bollinger = indicators['bollinger']
        ema_21 = indicators['ema']
        rsi = indicators['rsi']

        # LONG condition: close below lower band AND above EMA 21 AND RSI >50
        long_mask = (
            (df.close < bollinger['lower']) &
            (df.close > ema_21) &
            (rsi > 50)
        )
        signals[long_mask] = 1.0

        # SHORT condition: close above upper band AND below EMA 21 AND RSI <50
        short_mask = (
            (df.close > bollinger['upper']) &
            (df.close < ema_21) &
            (rsi < 50)
        )
        signals[short_mask] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_181105_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_181105_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self):
        return ['bollinger', 'rsi', 'ema']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # LONG conditions
        long_boll = df['close'] > df['bollinger'].upper
        long_rsi = df['rsi'] < 30
        long_ema = df['ema_fast'] > df['ema_slow']
        signals.loc[long_boll & long_rsi & long_ema] = 1.0

        # SHORT conditions
        short_boll = df['close'] < df['bollinger'].lower
        short_rsi = df['rsi'] > 70
        short_ema = df['ema_fast'] < df['ema_slow']
        signals.loc[short_boll & short_rsi & short_ema] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_181105_scalp_de_continuation_micro_retournemen\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_181105_scalp_de_continuation_micro_retournemen\strategy_v4.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                type="int",
                min=50,
                max=90,
                default=70,
                description="RSI overbought level"
            ),
            "rsi_oversold": ParameterSpec(
                type="int",
                min=10,
                max=50,
                default=30,
                description="RSI oversold level"
            ),
            "rsi_period": ParameterSpec(
                type="int",
                min=2,
                max=50,
                default=14,
                description="RSI period"
            ),
            "stop_atr_mult": ParameterSpec(
                type="float",
                min=1.0,
                max=3.0,
                default=1.5,
                description="Stop loss multiplier based on ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type="float",
                min=2.0,
                max=5.0,
                default=3.0,
                description="Take profit multiplier based on ATR"
            ),
            "warmup": ParameterSpec(
                type="int",
                min=20,
                max=100,
                default=50,
                description="Warmup period to ignore initial signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators with NaN protection
        bollinger = np.nan_to_num(indicators["bollinger"])
        rsi = np.nan_to_num(indicators["rsi"])
        ema = np.nan_to_num(indicators["ema"])

        for i in range(warmup, len(df)):
            # Current price conditions
            upper_band = bollinger[i, 0]
            lower_band = bollinger[i, 1]
            close = df.iloc[i].close

            # EMA condition
            ema_21 = ema[i, 1]  # Assuming ema[period=21] is second element

            # RSI conditions
            rsi_ob = params["rsi_overbought"]
            rsi_os = params["rsi_oversold"]

            # Check for long entry
            if (close > upper_band) and (rsi[i] < rsi_os) and (close > ema_21):
                signals.iloc[i] = 1.0  # LONG

            # Check for short entry
            elif (close < lower_band) and (rsi[i] > rsi_ob) and (close < ema_21):
                signals.iloc[i] = -1.0  # SHORT

            # Exit conditions
            elif (close < upper_band and signals.iloc[i-1] == 1.0) or \
                 (close > lower_band and signals.iloc[i-1] == -1.0):
                signals.iloc[i] = 0.0  # FLAT

            # Stop loss conditions based on Bollinger reversal
            elif ((signals.iloc[i-1] == 1.0 and close < lower_band) or
                  (signals.iloc[i-1] == -1.0 and close > upper_band)):
                signals.iloc[i] = 0.0  # FLAT

            # RSI divergence exit
            elif (rsi[i] < rsi_ob and signals.iloc[i-1] == -1.0) or \
                 (rsi[i] > rsi_os and signals.iloc[i-1] == 1.0):
                signals.iloc[i] = 0.0  # FLAT

        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_181856_x\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_181856_x\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='x')
    @property
    def required_indicators(self):
        return ['rsi','bollinger']
    @property
    def default_params(self):
        return {'a': 2, 'warmup': 50}
    def generate_signals(self, df, indicators, params):
        s = pd.Series(0.0, index=df.index)
        r = np.nan_to_num(indicators['rsi'])
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb['lower'])
        s[(r < 30) & (df['close'].values < lower)] = 1.0
        return s
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_181856_x\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_181856_x\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='x')
    @property
    def required_indicators(self):
        return ['rsi','bollinger']
    @property
    def default_params(self):
        return {'a':1}
    def generate_signals(self, df, indicators, params):
        s = pd.Series(0.0, index=df.index)
        r = np.nan_to_num(indicators['rsi'])
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb['lower'])
        s[(r < 30) & (df['close'].values < lower)] = 1.0
        return s
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_181856_x\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_181856_x\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase
class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='x')
    @property
    def required_indicators(self):
        return ['rsi','bollinger']
    @property
    def default_params(self):
        return {'a': 2, 'warmup': 50}
    def generate_signals(self, df, indicators, params):
        s = pd.Series(0.0, index=df.index)
        r = np.nan_to_num(indicators['rsi'])
        bb = indicators['bollinger']
        lower = np.nan_to_num(bb['lower'])
        s[(r < 30) & (df['close'].values < lower)] = 1.0
        return s
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_181944_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_181944_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(float, (0, 100)),
            "rsi_oversold": ParameterSpec(float, (0, 100)),
            "rsi_period": ParameterSpec(int, (1, 100)),
            "stop_atr_mult": ParameterSpec(float, (0.1, 5)),
            "tp_atr_mult": ParameterSpec(float, (0.1, 5)),
            "warmup": ParameterSpec(int, (10, 100))
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        ema_trend = np.nan_to_num(indicators["ema"])

        # Get ATR for stop and profit levels
        atr = np.nan_to_num(indicators["atr"])

        for i in range(warmup, len(df)):
            current_rsi = rsi[i]
            close_price = df.loc[i, "close"]

            # Determine trend direction
            ema_up = ema_trend[i] > 0
            ema_down = ema_trend[i] < 0

            # Long entry conditions
            if current_rsi > params["rsi_oversold"] and close_price < lower_bb[i]:
                if ema_up:
                    signals.iloc[i] = 1.0

                    # Take profit at upper BB or next support level
                    if close_price >= upper_bb[i]:
                        signals.iloc[i] = 0.0

                    # Stop loss below recent low
                    sl_level = close_price - atr[i] * params["stop_atr_mult"]
                    if close_price <= sl_level:
                        signals.iloc[i] = 0.0

            # Short entry conditions
            elif current_rsi < params["rsi_overbought"] and close_price > upper_bb[i]:
                if ema_down:
                    signals.iloc[i] = -1.0

                    # Take profit at lower BB or next resistance level
                    if close_price <= lower_bb[i]:
                        signals.iloc[i] = 0.0

                    # Stop loss above recent high
                    sl_level = close_price + atr[i] * params["stop_atr_mult"]
                    if close_price >= sl_level:
                        signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_181944_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_181944_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_period": 20,
            "ema_periods": [9, 21, 50],
            "rsi_period": 14,
            "std_dev": 2,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_period": ParameterSpec(int, min=5, max=50),
            "ema_periods": ParameterSpec(list, subtype=int, min=5, max=100),
            "rsi_period": ParameterSpec(int, min=2, max=50),
            "std_dev": ParameterSpec(float, min=1.0, max=3.0),
            "stop_atr_mult": ParameterSpec(float, min=1.0, max=2.0),
            "tp_atr_mult": ParameterSpec(float, min=2.0, max=4.0),
            "warmup": ParameterSpec(int, min=10, max=100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators
        ema_9 = np.nan_to_num(indicators["ema"][0])
        ema_21 = np.nan_to_num(indicators["ema"][1])
        ema_50 = np.nan_to_num(indicators["ema"][2])
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Initialize previous values for exit conditions
        prev_close = df.close.shift(1)
        prev_bb_upper = bb_upper.shift(1)
        prev_bb_lower = bb_lower.shift(1)

        for i in range(warmup, len(df)):
            current_close = df.close.iloc[i]
            current_rsi = rsi[i]
            current_bb_upper = bb_upper[i]
            current_bb_lower = bb_lower[i]

            # Entry conditions
            if (current_close > ema_21[i] and
                current_rsi > 50 and
                current_close < current_bb_upper):
                signals.iloc[i] = 1.0

            elif (current_close < ema_21[i] and
                  current_rsi < 50 and
                  current_close > current_bb_lower):
                signals.iloc[i] = -1.0

            else:
                # Exit conditions
                if ((current_close > prev_bb_upper and signals.iloc[i-1] == 1.0) or
                    (current_close < prev_bb_lower and signals.iloc[i-1] == -1.0)):
                    signals.iloc[i] = 0.0

                elif (rsi[i] < rsi[i-1] and signals.iloc[i-1] == 1.0 and current_rsi > 30):
                    signals.iloc[i] = 0.0

                elif (rsi[i] > rsi[i-1] and signals.iloc[i-1] == -1.0 and current_rsi < 70):
                    signals.iloc[i] = 0.0

                else:
                    signals.iloc[i] = signals.iloc[i-1]

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_181944_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_181944_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "bollinger_std_dev": 2,
            "ema_period": 21,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "bollinger_std_dev": ParameterSpec(
                type=float,
                bounds=(1.0, 3.0),
                default=2.0,
                description="Number of standard deviations for Bollinger Bands"
            ),
            "ema_period": ParameterSpec(
                type=int,
                bounds=(9, 50),
                default=21,
                description="Period for EMA calculation"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                bounds=(8, 21),
                default=14,
                description="Period for RSI calculation"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                bounds=(1.0, 3.0),
                default=1.5,
                description="Multiple of ATR for stop loss"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                bounds=(2.0, 4.0),
                default=3.0,
                description="Multiple of ATR for take profit"
            ),
            "warmup": ParameterSpec(
                type=int,
                bounds=(50, 100),
                default=50,
                description="Number of bars to skip for initial warmup"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period to avoid NaN signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators with NaN converted to 0
        ema_21 = np.nan_to_num(indicators["ema"][str(params["ema_period"])])
        rsi = np.nan_to_num(indicators["rsi"][str(params["rsi_period"])])
        bb = indicators["bollinger"]
        upper_bb = np.nan_to_num(bb["upper"])
        lower_bb = np.nan_to_num(bb["lower"])
        mid_bb = np.nan_to_num(bb["mid"])

        # Entry conditions
        long_entry = (
            (df.close > ema_21) &
            (rsi > 50) &
            (df.close < upper_bb)
        )

        short_entry = (
            (df.close < ema_21) &
            (rsi < 50) &
            (df.close > lower_bb)
        )

        # Exit conditions
        in_long = signals.shift(1).fillna(0).astype(int) == 1
        in_short = signals.shift(1).fillna(0).astype(int) == -1

        long_exit = (
            (df.close > upper_bb) |
            (rsi < 50)
        )

        short_exit = (
            (df.close < lower_bb) |
            (rsi > 50)
        )

        # Generate signals
        for i in range(warmup, len(df)):
            if long_entry[i]:
                signals.iloc[i] = 1.0
            elif short_entry[i]:
                signals.iloc[i] = -1.0
            elif in_long[i] and long_exit[i]:
                signals.iloc[i] = 0.0
            elif in_short[i] and short_exit[i]:
                signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_181944_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_181944_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(float, (0, 100)),
            "rsi_oversold": ParameterSpec(float, (0, 100)),
            "rsi_period": ParameterSpec(int, (1, 100)),
            "stop_atr_mult": ParameterSpec(float, (0.1, 5)),
            "tp_atr_mult": ParameterSpec(float, (0.1, 5)),
            "warmup": ParameterSpec(int, (10, 100))
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_bb = np.nan_to_num(bb["lower"])
        upper_bb = np.nan_to_num(bb["upper"])
        ema_trend = np.nan_to_num(indicators["ema"])

        # Get ATR for stop and profit levels
        atr = np.nan_to_num(indicators["atr"])

        for i in range(warmup, len(df)):
            current_rsi = rsi[i]
            close_price = df.loc[i, "close"]

            # Determine trend direction
            ema_up = ema_trend[i] > 0
            ema_down = ema_trend[i] < 0

            # Long entry conditions
            if current_rsi > params["rsi_oversold"] and close_price < lower_bb[i]:
                if ema_up:
                    signals.iloc[i] = 1.0

                    # Take profit at upper BB or next support level
                    if close_price >= upper_bb[i]:
                        signals.iloc[i] = 0.0

                    # Stop loss below recent low
                    sl_level = close_price - atr[i] * params["stop_atr_mult"]
                    if close_price <= sl_level:
                        signals.iloc[i] = 0.0

            # Short entry conditions
            elif current_rsi < params["rsi_overbought"] and close_price > upper_bb[i]:
                if ema_down:
                    signals.iloc[i] = -1.0

                    # Take profit at lower BB or next resistance level
                    if close_price <= lower_bb[i]:
                        signals.iloc[i] = 0.0

                    # Stop loss above recent high
                    sl_level = close_price + atr[i] * params["stop_atr_mult"]
                    if close_price >= sl_level:
                        signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_182712_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_182712_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_21_pullback_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"ema_period": 21, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_period": ParameterSpec(
                type=int,
                bounds=(2, 100),
                default=21,
                description="Period for EMA calculation"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                bounds=(2, 50),
                default=14,
                description="Period for RSI calculation"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                bounds=(0.5, 3),
                default=1.5,
                description="Stop loss multiplier for ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                bounds=(1, 4),
                default=3.0,
                description="Take profit multiplier for ATR"
            ),
            "warmup": ParameterSpec(
                type=int,
                bounds=(20, 100),
                default=50,
                description="Number of warmup bars to avoid initial NaN signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period to avoid NaN signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators with proper access
        ema = np.nan_to_num(indicators["ema"][params["ema_period"]])
        rsi = np.nan_to_num(indicators["rsi"][params["rsi_period"]])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])

        # Entry conditions
        long_entry = (df.close > ema) & (rsi < 70)
        short_entry = (df.close < ema) & (rsi > 30)

        # Exit conditions
        price_cross_ema = df.close < ema
        rsi_cross_70 = rsi > 70

        # Signal logic
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Close positions when exit conditions met
        signals[price_cross_ema] = 0.0
        signals[rsi_cross_70] = 0.0

        # Stop loss and take profit logic
        atr = np.nan_to_num(indicators["atr"])
        stop_level = df.close - params["stop_atr_mult"] * atr
        tp_level_long = df.close + params["tp_atr_mult"] * atr
        tp_level_short = df.close - params["tp_atr_mult"] * atr

        # Check for stop loss and take profit conditions
        signals[(df.close < stop_level)] = 0.0
        signals[(df.close > tp_level_long)] = 0.0
        signals[(df.close > stop_level)] = 0.0
        signals[(df.close < tp_level_short)] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_182712_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_182712_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_rsi_strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(type=float, bounds=(30, 70), default=70),
            "rsi_oversold": ParameterSpec(type=float, bounds=(30, 70), default=30),
            "rsi_period": ParameterSpec(type=int, bounds=(2, 100), default=14),
            "stop_atr_mult": ParameterSpec(type=float, bounds=(0.5, 2.0), default=1.5),
            "tp_atr_mult": ParameterSpec(type=float, bounds=(1.0, 4.0), default=3.0),
            "warmup": ParameterSpec(type=int, bounds=(20, 100), default=50)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        bollinger = indicators["bollinger"]
        bb_lower = np.nan_to_num(bollinger["lower"])
        bb_upper = np.nan_to_num(bollinger["upper"])
        bb_middle = np.nan_to_num(bollinger["middle"])

        ema = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])

        for i in range(warmup, len(df)):
            current_close = df.iloc[i]["close"]
            current_rsi = rsi[i]
            current_ema = ema[i]
            lower_band = bb_lower[i]
            upper_band = bb_upper[i]

            # Entry conditions
            if (current_close > current_ema and current_rsi > params["rsi_overbought"]) or \
               (current_close < current_ema and current_rsi < params["rsi_oversold"]):
                if (current_close > upper_band or current_close < lower_band):
                    signals.iloc[i] = 1.0 if current_close > current_ema else -1.0

            # Exit conditions
            if signals.iloc[i-1] != 0.0:
                if (current_close > upper_band or current_close < lower_band):
                    signals.iloc[i] = 0.0

                # Check for RSI divergence
                prev_rsi = rsi[i-1]
                if signals.iloc[i-1] == 1.0 and (prev_rsi < current_rsi):
                    signals.iloc[i] = 0.0
                elif signals.iloc[i-1] == -1.0 and (prev_rsi > current_rsi):
                    signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_182712_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_182712_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_rsi_strategy_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "ema", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14,
                "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(type=int, min=50, max=90),
            "rsi_oversold": ParameterSpec(type=int, min=10, max=40),
            "rsi_period": ParameterSpec(type=int, min=2, max=20),
            "stop_atr_mult": ParameterSpec(type=float, min=1.0, max=2.5),
            "tp_atr_mult": ParameterSpec(type=float, min=2.0, max=4.0),
            "warmup": ParameterSpec(type=int, min=20, max=100)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Get indicators with NaN handling
        bollinger = indicators["bollinger"]
        ema_9 = np.nan_to_num(indicators["ema"]["9"])
        ema_21 = np.nan_to_num(indicators["ema"]["21"])
        ema_50 = np.nan_to_num(indicators["ema"]["50"])
        rsi = np.nan_to_num(indicators["rsi"])
        close = np.nan_to_num(df["close"])

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        for i in range(warmup, len(df)):
            # LONG entries
            if (
                (close[i] > ema_21[i]) & (rsi[i] > params["rsi_overbought"]) or
                (close[i] > np.nan_to_num(bollinger["upper"])[i]) & (rsi[i] > params["rsi_overbought"] - 10)
            ):
                signals.iloc[i] = 1.0

            # SHORT entries
            elif (
                (close[i] < ema_21[i]) & (rsi[i] < params["rsi_oversold"]) or
                (close[i] < np.nan_to_num(bollinger["lower"])[i]) & (rsi[i] < params["rsi_oversold"] + 10)
            ):
                signals.iloc[i] = -1.0

            # Exit conditions
            if signals.iloc[i] != 0.0:
                # Check for opposite band exit
                if (
                    (signals.iloc[i] == 1.0 and close[i] < np.nan_to_num(bollinger["lower"])[i]) or
                    (signals.iloc[i] == -1.0 and close[i] > np.nan_to_num(bollinger["upper"])[i])
                ):
                    signals.iloc[i] = 0.0

                # Check for RSI divergence
                elif (
                    (signals.iloc[i] == 1.0 and rsi[i] < rsi[i-1]) or
                    (signals.iloc[i] == -1.0 and rsi[i] > rsi[i-1])
                ):
                    signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_182712_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_182712_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_ema")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                type=int,
                default=70,
                min=50,
                max=90,
                description="RSI overbought level"
            ),
            "rsi_oversold": ParameterSpec(
                type=int,
                default=30,
                min=10,
                max=70,
                description="RSI oversold level"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                default=14,
                min=5,
                max=20,
                description="RSI period"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                default=1.5,
                min=1.0,
                max=2.0,
                description="Stop loss multiple of ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                default=3.0,
                min=2.0,
                max=4.0,
                description="Take profit multiple of ATR"
            ),
            "warmup": ParameterSpec(
                type=int,
                default=50,
                min=20,
                max=100,
                description="Warmup period to filter initial signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period to avoid early signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators
        bollinger = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])
        ema = indicators["ema"]

        # Entry conditions
        long_entry = (
            df.close > np.nan_to_num(bollinger["upper"]) &
            rsi < params["rsi_oversold"] &
            ema["slow"] > ema["very_slow"]
        )

        short_entry = (
            df.close < np.nan_to_num(bollinger["lower"]) &
            rsi > params["rsi_overbought"] &
            ema["slow"] < ema["very_slow"]
        )

        # Exit conditions based on Bollinger bands
        long_exit = df.close < np.nan_to_num(bollinger["lower"])
        short_exit = df.close > np.nan_to_num(bollinger["upper"])

        # Update signals
        for i in range(warmup, len(df)):
            if long_entry.iloc[i]:
                signals.iloc[i] = 1.0
            elif short_entry.iloc[i]:
                signals.iloc[i] = -1.0

            # Exit conditions
            if signals.iloc[i] == 1.0 and (long_exit.iloc[i] or df.close.iloc[i] > np.nan_to_num(bollinger["upper"])):
                signals.iloc[i] = 0.0
            elif signals.iloc[i] == -1.0 and (short_exit.iloc[i] or df.close.iloc[i] < np.nan_to_num(bollinger["lower"])):
                signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_182712_scalp_de_continuation_micro_retournemen\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_182712_scalp_de_continuation_micro_retournemen\strategy_v4.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_21_pullback_rsi")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"ema_period": 21, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "ema_period": ParameterSpec(
                type=int,
                bounds=(2, 100),
                default=21,
                description="Period for EMA calculation"
            ),
            "rsi_period": ParameterSpec(
                type=int,
                bounds=(2, 50),
                default=14,
                description="Period for RSI calculation"
            ),
            "stop_atr_mult": ParameterSpec(
                type=float,
                bounds=(0.5, 3),
                default=1.5,
                description="Stop loss multiplier for ATR"
            ),
            "tp_atr_mult": ParameterSpec(
                type=float,
                bounds=(1, 4),
                default=3.0,
                description="Take profit multiplier for ATR"
            ),
            "warmup": ParameterSpec(
                type=int,
                bounds=(20, 100),
                default=50,
                description="Number of warmup bars to avoid initial NaN signals"
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup period to avoid NaN signals
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Get indicators with proper access
        ema = np.nan_to_num(indicators["ema"][params["ema_period"]])
        rsi = np.nan_to_num(indicators["rsi"][params["rsi_period"]])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])

        # Entry conditions
        long_entry = (df.close > ema) & (rsi < 70)
        short_entry = (df.close < ema) & (rsi > 30)

        # Exit conditions
        price_cross_ema = df.close < ema
        rsi_cross_70 = rsi > 70

        # Signal logic
        signals[long_entry] = 1.0
        signals[short_entry] = -1.0

        # Close positions when exit conditions met
        signals[price_cross_ema] = 0.0
        signals[rsi_cross_70] = 0.0

        # Stop loss and take profit logic
        atr = np.nan_to_num(indicators["atr"])
        stop_level = df.close - params["stop_atr_mult"] * atr
        tp_level_long = df.close + params["tp_atr_mult"] * atr
        tp_level_short = df.close - params["tp_atr_mult"] * atr

        # Check for stop loss and take profit conditions
        signals[(df.close < stop_level)] = 0.0
        signals[(df.close > tp_level_long)] = 0.0
        signals[(df.close > stop_level)] = 0.0
        signals[(df.close < tp_level_short)] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_184329_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_184329_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger_upper', 'bollinger_lower']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            rsi = indicators['rsi'].iloc[i]
            boll_upper = indicators['bollinger_upper'].iloc[i]
            boll_lower = indicators['bollinger_lower'].iloc[i]
            close_price = df.iloc[i]['close']

            if rsi < 30 and close_price < boll_lower:
                signals.iloc[i] = 1.0
            elif rsi > 70 and close_price > boll_upper:
                signals.iloc[i] = -1.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_184329_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_184329_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="BuilderStrategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"period": 14}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "period": ParameterSpec(
                type_=int,
                description="Period for RSI calculation.",
                constraints=(2, 50),
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        ema_21 = np.nan_to_num(indicators["ema"][("close", 21)])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        # logic here
        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_184329_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_184329_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from strategies.base import StrategyBase

class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="generated")

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger_upper', 'bollinger_lower']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, pd.Series], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        for i in range(len(df)):
            rsi = indicators['rsi'].iloc[i]
            boll_upper = indicators['bollinger_upper'].iloc[i]
            boll_lower = indicators['bollinger_lower'].iloc[i]
            close_price = df.iloc[i]['close']

            if rsi < 30 and close_price < boll_lower:
                signals.iloc[i] = 1.0
            elif rsi > 70 and close_price > boll_upper:
                signals.iloc[i] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_192500_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_192500_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_ema_rsi_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=40, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema9 = np.nan_to_num(indicators["ema"]["ema9"])
        ema21 = np.nan_to_num(indicators["ema"]["ema21"])
        ema50 = np.nan_to_num(indicators["ema"]["ema50"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])

        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)

        # Stop loss and take profit parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Warmup
        warmup = int(params.get("warmup", 50))

        # Entry conditions
        # Long entry: price < ema21, rsi > oversold, rsi crosses above oversold, price > lower bollinger band, close > ema21
        long_condition = (
            (df["close"].values < ema21) &
            (rsi > rsi_oversold) &
            (np.roll(rsi, 1) <= rsi_oversold) &
            (df["close"].values > bb_lower) &
            (df["close"].values > ema21)
        )

        # Short entry: price > ema21, rsi < overbought, rsi crosses below overbought, price < upper bollinger band, close < ema21
        short_condition = (
            (df["close"].values > ema21) &
            (rsi < rsi_overbought) &
            (np.roll(rsi, 1) >= rsi_overbought) &
            (df["close"].values < bb_upper) &
            (df["close"].values < ema21)
        )

        # Exit conditions
        # Exit long if close crosses above upper bollinger band or rsi crosses below oversold and price < ema21
        long_exit = (
            (df["close"].values > bb_upper) |
            ((rsi < rsi_oversold) & (df["close"].values < ema21))
        )

        # Exit short if close crosses below lower bollinger band or rsi crosses above overbought and price > ema21
        short_exit = (
            (df["close"].values < bb_lower) |
            ((rsi > rsi_overbought) & (df["close"].values > ema21))
        )

        # Generate signals
        long_signals = np.zeros_like(df["close"], dtype=float)
        short_signals = np.zeros_like(df["close"], dtype=float)

        # Initialize positions
        long_positions = np.zeros_like(df["close"], dtype=bool)
        short_positions = np.zeros_like(df["close"], dtype=bool)

        # Loop through data to apply logic
        for i in range(1, len(df)):
            if long_condition[i]:
                long_signals[i] = 1.0
                long_positions[i] = True
            elif short_condition[i]:
                short_signals[i] = -1.0
                short_positions[i] = True
            elif long_positions[i-1] and long_exit[i]:
                long_signals[i] = 0.0
                long_positions[i] = False
            elif short_positions[i-1] and short_exit[i]:
                short_signals[i] = 0.0
                short_positions[i] = False

        # Combine signals
        signals = pd.Series(long_signals - short_signals, index=df.index, dtype=np.float64)

        # Ensure warmup is set
        signals.iloc[:warmup] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_192500_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_192500_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr", "ema"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.5),
            "warmup": ParameterSpec(30, 100, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        middle = np.nan_to_num(bb["middle"])
        upper = np.nan_to_num(bb["upper"])
        ema_21 = np.nan_to_num(indicators["ema"]["ema_21"])
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Entry long conditions
        price_below_ema_21 = df["close"] < ema_21
        rsi_below_oversold = rsi < rsi_oversold
        rsi_rising = np.roll(rsi, 1) < rsi
        price_above_lower_bb = df["close"] > lower
        rsi_in_oversold = (rsi > rsi_oversold) & (rsi < rsi_oversold + 10)

        long_condition = (
            price_below_ema_21 &
            rsi_below_oversold &
            rsi_rising &
            price_above_lower_bb &
            rsi_in_oversold
        )

        # Entry short conditions
        price_above_ema_21 = df["close"] > ema_21
        rsi_above_overbought = rsi > rsi_overbought
        rsi_falling = np.roll(rsi, 1) > rsi
        price_below_upper_bb = df["close"] < upper
        rsi_in_overbought = (rsi < rsi_overbought) & (rsi > rsi_overbought - 10)

        short_condition = (
            price_above_ema_21 &
            rsi_above_overbought &
            rsi_falling &
            price_below_upper_bb &
            rsi_in_overbought
        )

        # Generate signals
        signals.loc[long_condition] = 1.0
        signals.loc[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_192500_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_192500_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_ema_rsi_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(10, 20, 1),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, 0.1),
            "warmup": ParameterSpec(30, 70, 1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        ema_9 = np.nan_to_num(indicators["ema"]["ema_9"])
        ema_21 = np.nan_to_num(indicators["ema"]["ema_21"])
        ema_50 = np.nan_to_num(indicators["ema"]["ema_50"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])

        price = np.nan_to_num(df["close"].values)
        prev_price = np.roll(price, 1)
        prev_rsi = np.roll(rsi, 1)

        # Entry long conditions
        long_condition = (
            (price > ema_21) &
            (prev_price <= ema_21) &
            (rsi > rsi_oversold) &
            (rsi > prev_rsi) &
            (price < bb_upper) &
            (price > bb_lower) &
            (prev_price <= bb_lower) &
            (price > prev_price)
        )

        # Entry short conditions
        short_condition = (
            (price < ema_21) &
            (prev_price >= ema_21) &
            (rsi < rsi_overbought) &
            (rsi < prev_rsi) &
            (price > bb_lower) &
            (price < bb_upper) &
            (prev_price >= bb_upper) &
            (price < prev_price)
        )

        # Generate signals
        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)

        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_192500_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_192500_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="btc_ema_rsi_bollinger_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=40, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=2),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema9 = np.nan_to_num(indicators["ema"]["ema9"])
        ema21 = np.nan_to_num(indicators["ema"]["ema21"])
        ema50 = np.nan_to_num(indicators["ema"]["ema50"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr = np.nan_to_num(indicators["atr"])

        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)

        # Stop loss and take profit parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Warmup
        warmup = int(params.get("warmup", 50))

        # Entry conditions
        # Long entry: price < ema21, rsi > oversold, rsi crosses above oversold, price > lower bollinger band, close > ema21
        long_condition = (
            (df["close"].values < ema21) &
            (rsi > rsi_oversold) &
            (np.roll(rsi, 1) <= rsi_oversold) &
            (df["close"].values > bb_lower) &
            (df["close"].values > ema21)
        )

        # Short entry: price > ema21, rsi < overbought, rsi crosses below overbought, price < upper bollinger band, close < ema21
        short_condition = (
            (df["close"].values > ema21) &
            (rsi < rsi_overbought) &
            (np.roll(rsi, 1) >= rsi_overbought) &
            (df["close"].values < bb_upper) &
            (df["close"].values < ema21)
        )

        # Exit conditions
        # Exit long if close crosses above upper bollinger band or rsi crosses below oversold and price < ema21
        long_exit = (
            (df["close"].values > bb_upper) |
            ((rsi < rsi_oversold) & (df["close"].values < ema21))
        )

        # Exit short if close crosses below lower bollinger band or rsi crosses above overbought and price > ema21
        short_exit = (
            (df["close"].values < bb_lower) |
            ((rsi > rsi_overbought) & (df["close"].values > ema21))
        )

        # Generate signals
        long_signals = np.zeros_like(df["close"], dtype=float)
        short_signals = np.zeros_like(df["close"], dtype=float)

        # Initialize positions
        long_positions = np.zeros_like(df["close"], dtype=bool)
        short_positions = np.zeros_like(df["close"], dtype=bool)

        # Loop through data to apply logic
        for i in range(1, len(df)):
            if long_condition[i]:
                long_signals[i] = 1.0
                long_positions[i] = True
            elif short_condition[i]:
                short_signals[i] = -1.0
                short_positions[i] = True
            elif long_positions[i-1] and long_exit[i