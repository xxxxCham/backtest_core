]:
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
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v4")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema_21 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        long_condition = (df["close"] < ema_21) & (df["close"] > bb_lower) & (rsi > params["rsi_oversold"]) & (df["close"] > df["open"])
        short_condition = (df["close"] > ema_21) & (df["close"] < bb_upper) & (rsi < params["rsi_overbought"]) & (df["close"] < df["open"])

        # Exit conditions
        long_exit = (df["close"] > bb_upper) | (rsi > params["rsi_overbought"])
        short_exit = (df["close"] < bb_lower) | (rsi < params["rsi_oversold"])

        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 40, 1, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(5, 30, 1, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, 1, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper = np.nan_to_num(bb["upper"])
        middle = np.nan_to_num(bb["middle"])
        lower = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])

        # Get parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))

        # Initialize positions
        position = 0
        entry_price = 0.0
        stop_loss = 0.0
        take_profit = 0.0

        # Warmup
        signals.iloc[:warmup] = 0.0

        # Loop through data
        for i in range(warmup, len(df)):
            # Entry conditions
            price = df["close"].iloc[i]
            prev_rsi = rsi[i-1] if i > 0 else 0

            # Long entry: price crosses above lower Bollinger Band, RSI below 30, RSI rising
            if (price > lower[i] and
                rsi[i] < rsi_oversold and
                rsi[i] > prev_rsi):
                if position == 0:
                    position = 1
                    entry_price = price
                    stop_loss = entry_price - stop_atr_mult * atr[i]
                    take_profit = entry_price + tp_atr_mult * atr[i]
                    signals.iloc[i] = 1.0  # LONG signal

            # Short entry: price crosses below upper Bollinger Band, RSI above 70, RSI falling
            elif (price < upper[i] and
                  rsi[i] > rsi_overbought and
                  rsi[i] < prev_rsi):
                if position == 0:
                    position = -1
                    entry_price = price
                    stop_loss = entry_price + stop_atr_mult * atr[i]
                    take_profit = entry_price - tp_atr_mult * atr[i]
                    signals.iloc[i] = -1.0  # SHORT signal

            # Exit conditions
            elif position == 1:  # Long position
                if (price >= take_profit or
                    price <= stop_loss or
                    rsi[i] > rsi_overbought or
                    rsi[i] < rsi_oversold):
                    position = 0
                    signals.iloc[i] = 0.0  # FLAT signal

            elif position == -1:  # Short position
                if (price <= take_profit or
                    price >= stop_loss or
                    rsi[i] > rsi_overbought or
                    rsi[i] < rsi_oversold):
                    position = 0
                    signals.iloc[i] = 0.0  # FLAT signal

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

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
            "warmup": ParameterSpec(30, 70, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        bb_upper = np.nan_to_num(bb["upper"])
        bb_lower = np.nan_to_num(bb["lower"])
        bb_middle = np.nan_to_num(bb["middle"])
        atr = np.nan_to_num(indicators["atr"])

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Prepare arrays for comparison
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Entry conditions
        # Long entry: price crosses above EMA 21, RSI below oversold and rising, price above lower BB
        ema_21 = ema_9  # Using EMA 9 as proxy for EMA 21 since we only have EMA 9
        price = df["close"].values

        # Previous EMA values
        ema_21_prev = np.roll(ema_21, 1)
        rsi_prev = np.roll(rsi, 1)
        price_prev = np.roll(price, 1)
        bb_lower_prev = np.roll(bb_lower, 1)
        bb_upper_prev = np.roll(bb_upper, 1)

        # Long signal conditions
        long_condition = (
            (price > ema_21) & (price_prev <= ema_21_prev) &
            (rsi < rsi_oversold) & (rsi > rsi_prev) &
            (price > bb_lower)
        )

        # Short signal conditions
        short_condition = (
            (price < ema_21) & (price_prev >= ema_21_prev) &
            (rsi > rsi_overbought) & (rsi < rsi_prev) &
            (price < bb_upper)
        )

        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract and sanitize indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        ema_21 = np.nan_to_num(indicators["ema"])
        ema_50 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])

        # Use specific EMA values (assuming ema array is ordered by period)
        # For this strategy, we'll use ema_9, ema_21, ema_50 from the ema array
        # We'll assume ema array is sorted with 9, 21, 50 period EMA in that order
        ema_9 = ema_9[0] if isinstance(ema_9, np.ndarray) and len(ema_9) > 0 else np.full_like(rsi, 0)
        ema_21 = ema_21[1] if isinstance(ema_21, np.ndarray) and len(ema_21) > 1 else np.full_like(rsi, 0)
        ema_50 = ema_50[2] if isinstance(ema_50, np.ndarray) and len(ema_50) > 2 else np.full_like(rsi, 0)

        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        # Stop loss and take profit multipliers
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Warmup
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Create condition arrays
        price = df["close"].values
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]

        # Long entry conditions
        long_condition_1 = price > ema_21
        long_condition_2 = price > ema_9
        long_condition_3 = rsi < rsi_oversold
        long_condition_4 = rsi > rsi_prev
        long_condition_5 = price > bb_lower
        long_condition_6 = price < bb_middle

        long_entry = long_condition_1 & long_condition_2 & long_condition_3 & long_condition_4 & long_condition_5 & long_condition_6

        # Short entry conditions
        short_condition_1 = price < ema_21
        short_condition_2 = price < ema_9
        short_condition_3 = rsi > rsi_overbought
        short_condition_4 = rsi < rsi_prev
        short_condition_5 = price < bb_upper
        short_condition_6 = price > bb_middle

        short_entry = short_condition_1 & short_condition_2 & short_condition_3 & short_condition_4 & short_condition_5 & short_condition_6

        # Exit conditions
        exit_long = (price > bb_upper) | (price < bb_lower) | ((rsi > 70) & (rsi < rsi_prev)) | ((rsi < 30) & (rsi > rsi_prev))
        exit_short = (price < bb_lower) | (price > bb_upper) | ((rsi > 70) & (rsi < rsi_prev)) | ((rsi < 30) & (rsi > rsi_prev))

        # Generate signals
        long_signals = np.where(long_entry, 1.0, 0.0)
        short_signals = np.where(short_entry, -1.0, 0.0)

        # Combine signals
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)

        # Ensure no conflicting signals at the same time
        # If both long and short signals are active, prefer long (or use a rule to resolve)
        # In this case, we'll just keep the first valid signal, so we'll apply the signals in order

        # Apply warmup protection
        signals.iloc[:warmup] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v4.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=60, max_value=80, step=5),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=10, max_value=30, step=5),
            "rsi_period": ParameterSpec(param_type="int", min_value=10, max_value=20, step=5),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=2.0, step=0.25),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=2.0, max_value=4.0, step=0.5),
            "warmup": ParameterSpec(param_type="int", min_value=30, max_value=70, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema_9 = np.nan_to_num(indicators["ema"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])

        # Price array
        price = np.nan_to_num(df["close"].values)

        # RSI parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)

        # ATR parameters
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Compute previous RSI for divergence
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = rsi[0]

        # Entry conditions
        long_condition = (
            (price < bb_middle) &
            (price > bb_lower) &
            (price > ema_9) &
            (rsi < rsi_oversold) &
            (rsi > rsi_prev)
        )

        short_condition = (
            (price > bb_middle) &
            (price < bb_upper) &
            (price < ema_9) &
            (rsi > rsi_overbought) &
            (rsi < rsi_prev)
        )

        # Exit conditions
        long_exit = (
            (price > bb_upper) |
            ((rsi > rsi_overbought) & (rsi < rsi_prev)) |
            ((rsi < rsi_oversold) & (rsi > rsi_prev))
        )

        short_exit = (
            (price < bb_lower) |
            ((rsi > rsi_overbought) & (rsi < rsi_prev)) |
            ((rsi < rsi_oversold) & (rsi > rsi_prev))
        )

        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy_v5.py -->
```json
{
  "name": "strategy_v5.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v5.py",
  "ext": ".py",
  "anchor": "strategy_v5_py"
}
```
## strategy_v5_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v5.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 80, 1),
            "rsi_oversold": ParameterSpec(20, 40, 1),
            "rsi_period": ParameterSpec(5, 30, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
            "warmup": ParameterSpec(20, 100, 5),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema_21 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_middle = np.nan_to_num(indicators["bollinger"]["middle"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])

        # Previous RSI for divergence confirmation
        rsi_prev = np.roll(rsi, 1)

        # Entry conditions
        long_condition = (df["close"] > ema_21) & (df["close"] < bb_upper) & (rsi < params["rsi_oversold"]) & (rsi > rsi_prev)
        short_condition = (df["close"] < ema_21) & (df["close"] > bb_lower) & (rsi > params["rsi_overbought"]) & (rsi < rsi_prev)

        # Exit conditions
        exit_long = (df["close"] > bb_upper) | ((rsi > params["rsi_overbought"]) & (rsi < rsi_prev))
        exit_short = (df["close"] < bb_lower) | ((rsi < params["rsi_oversold"]) & (rsi > rsi_prev))

        # Generate signals
        long_entries = long_condition
        short_entries = short_condition

        # Set initial signals
        signals[long_entries] = 1.0
        signals[short_entries] = -1.0

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v5.py -->

<!-- MODULE-START: strategy_v6.py -->
```json
{
  "name": "strategy_v6.py",
  "path": "20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\\strategy_v6.py",
  "ext": ".py",
  "anchor": "strategy_v6_py"
}
```
## strategy_v6_py
*Path*: `20260211_193252_19_24_49_info_agents_ollama_manager_d_ma\strategy_v6.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_bollinger_rsi_scalper_v4")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(60, 90, 1),
            "rsi_oversold": ParameterSpec(10, 40, 1),
            "stop_atr_mult": ParameterSpec(1.0, 3.0, 0.1),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, 0.1),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract indicators
        ema_21 = np.nan_to_num(indicators["ema"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])

        # Warmup period
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Entry conditions
        long_condition = (df["close"] < ema_21) & (df["close"] > bb_lower) & (rsi > params["rsi_oversold"]) & (df["close"] > df["open"])
        short_condition = (df["close"] > ema_21) & (df["close"] < bb_upper) & (rsi < params["rsi_overbought"]) & (df["close"] < df["open"])

        # Exit conditions
        long_exit = (df["close"] > bb_upper) | (rsi > params["rsi_overbought"])
        short_exit = (df["close"] < bb_lower) | (rsi < params["rsi_oversold"])

        # Set signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v6.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy.py`  
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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Extract indicator arrays
        rsi = indicators['rsi']
        upper_band = indicators['bollinger']['upper']
        middle_band = indicators['bollinger']['middle']
        lower_band = indicators['bollinger']['lower']
        close = df['close'].values

        # Create boolean masks for long and short conditions
        long_condition = (rsi < 30) & (close < lower_band)
        short_condition = (rsi > 70) & (close > upper_band)

        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(name="rsi_overbought", param_type="int", min_value=50, max_value=90, step=5),
            "rsi_oversold": ParameterSpec(name="rsi_oversold", param_type="int", min_value=10, max_value=50, step=5),
            "rsi_period": ParameterSpec(name="rsi_period", param_type="int", min_value=5, max_value=30, step=5),
            "stop_atr_mult": ParameterSpec(name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.5),
            "tp_atr_mult": ParameterSpec(name="tp_atr_mult", param_type="float", min_value=2.0, max_value=5.0, step=0.5),
            "warmup": ParameterSpec(name="warmup", param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower_band = np.nan_to_num(bb["lower"])
        upper_band = np.nan_to_num(bb["upper"])
        price = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0
        long_conditions = (price < lower_band) & (rsi < rsi_oversold)
        short_conditions = (price > upper_band) & (rsi > rsi_overbought)
        long_signals = np.where(long_conditions, 1.0, 0.0)
        short_signals = np.where(short_conditions, -1.0, 0.0)
        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)
        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v10.py -->
```json
{
  "name": "strategy_v10.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v10.py",
  "ext": ".py",
  "anchor": "strategy_v10_py"
}
```
## strategy_v10_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v10.py`  
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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Extract indicator arrays
        rsi = indicators['rsi']
        upper_band = indicators['bollinger']['upper']
        middle_band = indicators['bollinger']['middle']
        lower_band = indicators['bollinger']['lower']
        close = df['close'].values

        # Create boolean masks for long and short conditions
        long_condition = (rsi < 30) & (close < lower_band)
        short_condition = (rsi > 70) & (close > upper_band)

        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v10.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v2.py`  
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
        return ['rsi', 'bollinger', 'atr', 'adx']

    @property
    def default_params(self):
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 20
        }

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Extract parameters
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        adx_threshold = params.get('adx_threshold', 20)

        # Extract indicators
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper_band = bollinger['upper']
        middle_band = bollinger['middle']
        lower_band = bollinger['lower']
        adx = indicators['adx']
        price = df['close'].values

        # Create masks for long and short conditions
        long_condition = (price < lower_band) & (rsi < rsi_oversold) & (adx > adx_threshold)
        short_condition = (price > upper_band) & (rsi > rsi_overbought) & (adx > adx_threshold)

        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(param_type="int", min_value=50, max_value=100, step=1),
            "rsi_oversold": ParameterSpec(param_type="int", min_value=0, max_value=50, step=1),
            "rsi_period": ParameterSpec(param_type="int", min_value=5, max_value=30, step=1),
            "stop_atr_mult": ParameterSpec(param_type="float", min_value=0.5, max_value=5.0, step=0.1),
            "tp_atr_mult": ParameterSpec(param_type="float", min_value=1.0, max_value=10.0, step=0.1),
            "warmup": ParameterSpec(param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        price = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))

        signals.iloc[:warmup] = 0.0

        long_condition = (price < lower) & (rsi < rsi_oversold)
        short_condition = (price > upper) & (rsi > rsi_overbought)

        long_signals = np.where(long_condition, 1.0, 0.0)
        short_signals = np.where(short_condition, -1.0, 0.0)

        signals = pd.Series(long_signals + short_signals, index=df.index, dtype=np.float64)

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v4.py`  
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
        return ['rsi', 'bollinger', 'atr', 'adx']

    @property
    def default_params(self):
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 20
        }

    def generate_signals(self, df, indicators, params):
        # Extract parameters with defaults
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        adx_threshold = params.get('adx_threshold', 20)

        # Extract indicator arrays
        price = df['close'].values
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper = bollinger['upper']
        middle = bollinger['middle']
        lower = bollinger['lower']
        adx = indicators['adx']

        # Create signals array
        signals = pd.Series(0.0, index=df.index)

        # Generate LONG signals
        long_condition = (price < lower) & (rsi < rsi_oversold) & (adx > adx_threshold)
        signals[long_condition] = 1.0

        # Generate SHORT signals
        short_condition = (price > upper) & (rsi > rsi_overbought) & (adx > adx_threshold)
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy_v5.py -->
```json
{
  "name": "strategy_v5.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v5.py",
  "ext": ".py",
  "anchor": "strategy_v5_py"
}
```
## strategy_v5_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v5.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(name="rsi_overbought", param_type="int", min_value=50, max_value=90, step=1),
            "rsi_oversold": ParameterSpec(name="rsi_oversold", param_type="int", min_value=10, max_value=50, step=1),
            "rsi_period": ParameterSpec(name="rsi_period", param_type="int", min_value=5, max_value=30, step=1),
            "stop_atr_mult": ParameterSpec(name="stop_atr_mult", param_type="float", min_value=1.0, max_value=3.0, step=0.1),
            "tp_atr_mult": ParameterSpec(name="tp_atr_mult", param_type="float", min_value=2.0, max_value=5.0, step=0.1),
            "warmup": ParameterSpec(name="warmup", param_type="int", min_value=20, max_value=100, step=10),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        lower = np.nan_to_num(bb["lower"])
        upper = np.nan_to_num(bb["upper"])
        price = np.nan_to_num(df["close"].values)
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)
        warmup = int(params.get("warmup", 50))

        signals.iloc[:warmup] = 0.0

        long_condition = (price < lower) & (rsi < rsi_oversold)
        short_condition = (price > upper) & (rsi > rsi_overbought)

        long_entries = np.where(long_condition, 1.0, 0.0)
        short_entries = np.where(short_condition, -1.0, 0.0)

        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        signals.iloc[:warmup] = 0.0

        positions = 0
        entry_price = 0.0
        for i in range(warmup, len(signals)):
            if positions == 0:
                if long_entries[i] == 1.0:
                    signals.iloc[i] = 1.0
                    positions = 1
                    entry_price = price[i]
                elif short_entries[i] == -1.0:
                    signals.iloc[i] = -1.0
                    positions = -1
                    entry_price = price[i]
            else:
                stop_loss = entry_price - (stop_atr_mult * atr[i])
                take_profit = entry_price + (tp_atr_mult * atr[i])
                if positions == 1:
                    if price[i] >= take_profit or price[i] <= stop_loss:
                        signals.iloc[i] = 0.0
                        positions = 0
                elif positions == -1:
                    if price[i] <= take_profit or price[i] >= stop_loss:
                        signals.iloc[i] = 0.0
                        positions = 0

        return signals
```
<!-- MODULE-END: strategy_v5.py -->

<!-- MODULE-START: strategy_v6.py -->
```json
{
  "name": "strategy_v6.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v6.py",
  "ext": ".py",
  "anchor": "strategy_v6_py"
}
```
## strategy_v6_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v6.py`  
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
        return ['rsi', 'bollinger', 'atr', 'adx']

    @property
    def default_params(self):
        return {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'adx_threshold': 20
        }

    def generate_signals(self, df, indicators, params):
        rsi_oversold = params.get('rsi_oversold', 30)
        rsi_overbought = params.get('rsi_overbought', 70)
        adx_threshold = params.get('adx_threshold', 20)

        signals = pd.Series(0.0, index=df.index)

        # Extract indicators
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper = bollinger['upper']
        middle = bollinger['middle']
        lower = bollinger['lower']
        adx = indicators['adx']
        price = df['close'].values

        # Create masks for long and short conditions
        long_condition = (price < lower) & (rsi < rsi_oversold) & (adx > adx_threshold)
        short_condition = (price > upper) & (rsi > rsi_overbought) & (adx > adx_threshold)

        # Generate signals
        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v6.py -->

<!-- MODULE-START: strategy_v7.py -->
```json
{
  "name": "strategy_v7.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v7.py",
  "ext": ".py",
  "anchor": "strategy_v7_py"
}
```
## strategy_v7_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v7.py`  
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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)

        # Extract indicator values
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        upper_band = bollinger['upper']
        lower_band = bollinger['lower']
        middle_band = bollinger['middle']
        price = df['close'].values

        # Create long signals
        long_condition = (price < lower_band) & (rsi < 30)
        signals[long_condition] = 1.0

        # Create short signals
        short_condition = (price > upper_band) & (rsi > 70)
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v7.py -->

<!-- MODULE-START: strategy_v8.py -->
```json
{
  "name": "strategy_v8.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v8.py",
  "ext": ".py",
  "anchor": "strategy_v8_py"
}
```
## strategy_v8_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v8.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="rsi_bollinger_atr_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"atr_threshold": 20, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "atr_threshold": ParameterSpec("atr_threshold", 10, 50, 20),
            "rsi_overbought": ParameterSpec("rsi_overbought", 60, 85, 70),
            "rsi_oversold": ParameterSpec("rsi_oversold", 15, 40, 30),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 14),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 1.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 6.0, 3.0),
            "warmup": ParameterSpec("warmup", 20, 100, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        rsi = np.nan_to_num(indicators["rsi"])
        bb = indicators["bollinger"]
        upper_band = np.nan_to_num(bb["upper"])
        middle_band = np.nan_to_num(bb["middle"])
        lower_band = np.nan_to_num(bb["lower"])
        atr = np.nan_to_num(indicators["atr"])

        # Parameters
        atr_threshold = params.get("atr_threshold", 20)
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        # Entry conditions
        close = np.nan_to_num(df["close"])
        long_condition = (close < lower_band) & (rsi < rsi_oversold) & (atr > atr_threshold)
        short_condition = (close > upper_band) & (rsi > rsi_overbought) & (atr > atr_threshold)

        # Exit conditions
        position = np.zeros_like(close)
        for i in range(1, len(df)):
            if position[i-1] != 0:
                if position[i-1] == 1 and close[i] > middle_band[i]:
                    position[i] = 0
                elif position[i-1] == -1 and close[i] < middle_band[i]:
                    position[i] = 0
                else:
                    position[i] = position[i-1]
            else:
                if long_condition[i]:
                    position[i] = 1
                elif short_condition[i]:
                    position[i] = -1
                else:
                    position[i] = 0

        # Generate signals
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        for i in range(1, len(df)):
            if position[i] == 1 and position[i-1] != 1:
                signals.iloc[i] = 1.0
            elif position[i] == -1 and position[i-1] != -1:
                signals.iloc[i] = -1.0
            elif position[i] == 0 and position[i-1] != 0:
                signals.iloc[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v8.py -->

<!-- MODULE-START: strategy_v9.py -->
```json
{
  "name": "strategy_v9.py",
  "path": "20260211_195443_warning_data_loader_plus_gros_gap_2019\\strategy_v9.py",
  "ext": ".py",
  "anchor": "strategy_v9_py"
}
```
## strategy_v9_py
*Path*: `20260211_195443_warning_data_loader_plus_gros_gap_2019\strategy_v9.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="supertrend_rsi_filter")

    @property
    def required_indicators(self) -> List[str]:
        return ["supertrend", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 60, "rsi_oversold": 40, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec("rsi_overbought", 50, 80, 1, 60),
            "rsi_oversold": ParameterSpec("rsi_oversold", 20, 50, 1, 40),
            "rsi_period": ParameterSpec("rsi_period", 5, 30, 1, 14),
            "stop_atr_mult": ParameterSpec("stop_atr_mult", 1.0, 3.0, 0.1, 1.5),
            "tp_atr_mult": ParameterSpec("tp_atr_mult", 2.0, 5.0, 0.1, 3.0),
            "warmup": ParameterSpec("warmup", 20, 100, 1, 50),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Extract indicators
        supertrend = indicators["supertrend"]
        rsi = np.nan_to_num(indicators["rsi"])
        atr = np.nan_to_num(indicators["atr"])
        close = np.nan_to_num(df["close"].values)

        # Supertrend bands
        upper_band = np.nan_to_num(supertrend["upper"])
        lower_band = np.nan_to_num(supertrend["lower"])

        # Entry conditions
        rsi_overbought = params.get("rsi_overbought", 60)
        rsi_oversold = params.get("rsi_oversold", 40)

        # Entry long: price crosses above upper band AND RSI is below oversold
        long_condition = (close > upper_band) & (rsi < rsi_oversold)

        # Entry short: price crosses below lower band AND RSI is above overbought
        short_condition = (close < lower_band) & (rsi > rsi_overbought)

        # Exit conditions
        # Exit long: price crosses below lower band
        exit_long_condition = close < lower_band

        # Exit short: price crosses above upper band
        exit_short_condition = close > upper_band

        # Initialize positions
        position = 0
        position_change = 0

        for i in range(warmup, len(signals)):
            if position == 0:
                if long_condition[i]:
                    signals[i] = 1.0
                    position = 1
                elif short_condition[i]:
                    signals[i] = -1.0
                    position = -1
            elif position == 1:
                if exit_long_condition[i]:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                if exit_short_condition[i]:
                    signals[i] = 0.0
                    position = 0

        return signals
```
<!-- MODULE-END: strategy_v9.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, int, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        ema = np.nan_to_num(indicators["ema"])
        bollinger = indicators["bollinger"]
        upper = np.nan_to_num(bollinger["upper"])
        lower = np.nan_to_num(bollinger["lower"])
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = int(params.get("rsi_overbought", 70))
        rsi_oversold = int(params.get("rsi_oversold", 30))

        long_condition = (df["close"] < ema) & (df["close"] <= lower) & (rsi < rsi_oversold)
        short_condition = (df["close"] > ema) & (df["close"] >= upper) & (rsi > rsi_overbought)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        exit_condition = (df["close"] >= upper) | (df["close"] <= lower) | (rsi > rsi_overbought) | (rsi < rsi_oversold)
        signals[exit_condition] = 0.0

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(20, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, int, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        long_condition = (
            bollinger["lower"] >= df["close"]
            & (rsi <= rsi_oversold)
            & (rsi > np.roll(rsi, 1))  # RSI increasing
        )

        short_condition = (
            bollinger["upper"] <= df["close"]
            & (rsi >= rsi_overbought)
            & (rsi < np.roll(rsi, 1))  # RSI decreasing
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v10.py -->
```json
{
  "name": "strategy_v10.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v10.py",
  "ext": ".py",
  "anchor": "strategy_v10_py"
}
```
## strategy_v10_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v10.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_ema_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "bollinger", "rsi", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(20, 100, int, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        ema = np.nan_to_num(indicators["ema"])
        bollinger = indicators["bollinger"]
        upper = np.nan_to_num(bollinger["upper"])
        lower = np.nan_to_num(bollinger["lower"])
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = int(params.get("rsi_overbought", 70))
        rsi_oversold = int(params.get("rsi_oversold", 30))

        long_condition = (df["close"] < ema) & (df["close"] <= lower) & (rsi < rsi_oversold)
        short_condition = (df["close"] > ema) & (df["close"] >= upper) & (rsi > rsi_overbought)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        exit_condition = (df["close"] >= upper) | (df["close"] <= lower) | (rsi > rsi_overbought) | (rsi < rsi_oversold)
        signals[exit_condition] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v10.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss Multiplier (ATR)"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit Multiplier (ATR)"),
            "warmup": ParameterSpec(20, 100, int, "Warmup Period"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper = np.nan_to_num(bollinger["upper"])
        lower = np.nan_to_num(bollinger["lower"])
        atr = np.nan_to_num(indicators["atr"])

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        long_condition = (df["close"] < lower) & (rsi < rsi_oversold)
        short_condition = (df["close"] > upper) & (rsi > rsi_overbought)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="ema_rsi_bollinger_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 60, 80, "Integer", "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(30, 20, 40, "Integer", "RSI Oversold Level"),
            "rsi_period": ParameterSpec(14, 5, 21, "Integer", "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.5, 0.5, 3.0, "Float", "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(3.0, 1.5, 5.0, "Float", "Take Profit ATR Multiplier"),
            "warmup": ParameterSpec(50, 20, 100, "Integer", "Warmup Period")
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi_overbought = int(params.get("rsi_overbought", 70))
        rsi_oversold = int(params.get("rsi_oversold", 30))

        ema_val = np.nan_to_num(indicators["ema"])
        rsi_val = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])
        atr_val = np.nan_to_num(indicators["atr"])

        long_condition = (df["close"] < ema_val) & (rsi_val < 50) & (rsi_val > rsi_oversold) & (df["close"] <= lower_band)
        short_condition = (df["close"] > ema_val) & (rsi_val > 50) & (rsi_val < rsi_overbought) & (df["close"] >= upper_band)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v4.py`  
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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df, indicators, params):
        signals = pd.Series(0.0, index=df.index)
        rsi = indicators['rsi']
        bollinger = indicators['bollinger']
        atr = indicators['atr']

        for i in range(1, len(df)):
            if (df['close'][i] < bollinger['lower'][i]) & (rsi[i] < 30) & (rsi[i] > rsi[i-1]):
                signals[i] = 1.0  # LONG
            elif (df['close'][i] > bollinger['upper'][i]) & (rsi[i] > 70) & (rsi[i] < rsi[i-1]):
                signals[i] = -1.0  # SHORT
        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy_v7.py -->
```json
{
  "name": "strategy_v7.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v7.py",
  "ext": ".py",
  "anchor": "strategy_v7_py"
}
```
## strategy_v7_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v7.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(20, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 5.0, float, "Take Profit ATR Multiplier"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        rsi_period = params.get("rsi_period", 14)
        stop_atr_mult = params.get("stop_atr_mult", 1.5)
        tp_atr_mult = params.get("tp_atr_mult", 3.0)

        rsi_values = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        atr_values = np.nan_to_num(indicators["atr"])

        long_condition = (
            (bollinger["lower"].astype(bool)) &
            (rsi_values > rsi_oversold) &
            (rsi_values < 50)
        )
        short_condition = (
            (bollinger["upper"].astype(bool)) &
            (rsi_values < rsi_overbought) &
            (rsi_values > 50)
        )

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v7.py -->

<!-- MODULE-START: strategy_v9.py -->
```json
{
  "name": "strategy_v9.py",
  "path": "20260211_201713_scalp_de_continuation_micro_retournemen\\strategy_v9.py",
  "ext": ".py",
  "anchor": "strategy_v9_py"
}
```
## strategy_v9_py
*Path*: `20260211_201713_scalp_de_continuation_micro_retournemen\strategy_v9.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_scalp")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(70, 90, float, "RSI Overbought Level"),
            "rsi_oversold": ParameterSpec(10, 30, float, "RSI Oversold Level"),
            "rsi_period": ParameterSpec(10, 20, int, "RSI Period"),
            "stop_atr_mult": ParameterSpec(1.0, 2.0, float, "Stop Loss ATR Multiplier"),
            "tp_atr_mult": ParameterSpec(2.0, 4.0, float, "Take Profit ATR Multiplier"),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)
        atr = np.nan_to_num(indicators["atr"])
        rsi_value = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_band = np.nan_to_num(bollinger["upper"])
        lower_band = np.nan_to_num(bollinger["lower"])

        long_condition = (lower_band == df["close"]) & (rsi_value < rsi_oversold)
        short_condition = (upper_band == df["close"]) & (rsi_value > rsi_overbought)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v9.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_patterns_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_width_threshold': 0.0005,
         'ema_long_period': 50,
         'ema_short_period': 21,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup_periods': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_momentum")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14, 'ema_periods': [21, 50], 'momentum_period': 10, 'warmup': 20}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v10.py -->
```json
{
  "name": "strategy_v10.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v10.py",
  "ext": ".py",
  "anchor": "strategy_v10_py"
}
```
## strategy_v10_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v10.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_patterns_v3")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr', 'macd']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_width_threshold': 0.0005,
         'ema_long_period': 50,
         'ema_short_period': 21,
         'macd_fast_period': 12,
         'macd_signal_period': 9,
         'macd_slow_period': 26,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup_periods': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v10.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v2.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_momentum_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'ema_long_period': 21,
         'ema_short_period': 12,
         'momentum_multiplier': 0.5,
         'momentum_period': 10,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 14,
         'stop_atr_mult': 1.2,
         'tp_atr_mult': 2.5,
         'volatility_squeeze_threshold': 0.1,
         'warmup': 30}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_momentum_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ["ema", "atr", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "momentum_multiplier": 0.5,
            "momentum_period": 10,
            "ema_long_period": 21,
            "ema_short_period": 12,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.2,
            "tp_atr_mult": 2.5,
            "volatility_squeeze_threshold": 0.1,
            "warmup": 30
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "momentum_multiplier": ParameterSpec(type_=(float, int), default=0.5, min_val=0.1, max_val=5.0),
            "momentum_period": ParameterSpec(type_=int, default=10, min_val=1, max_val=100),
            "ema_long_period": ParameterSpec(type_=int, default=21, min_val=2, max_val=100),
            "ema_short_period": ParameterSpec(type_=int, default=12, min_val=1, max_val=20),
            "rsi_overbought": ParameterSpec(type_=(int, float), default=70, min_val=1, max_val=100),
            "rsi_oversold": ParameterSpec(type_=(int, float), default=30, min_val=0, max_val=99),
            "rsi_period": ParameterSpec(type_=int, default=14, min_val=1, max_val=100),
            "stop_atr_mult": ParameterSpec(type_=(float, int), default=1.2, min_val=0.1, max_val=10.0),
            "tp_atr_mult": ParameterSpec(type_=(float, int), default=2.5, min_val=0.1, max_val=10.0),
            "volatility_squeeze_threshold": ParameterSpec(type_=(float, int), default=0.1, min_val=0.01, max_val=1.0),
            "warmup": ParameterSpec(type_=int, default=30, min_val=0, max_val=1000)
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Extract parameters
        momentum_multiplier = params["momentum_multiplier"]
        momentum_period = params["momentum_period"]
        ema_long_period = params["ema_long_period"]
        ema_short_period = params["ema_short_period"]
        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        rsi_period = params["rsi_period"]
        stop_atr_mult = params["stop_atr_mult"]
        tp_atr_mult = params["tp_atr_mult"]
        volatility_squeeze_threshold = params["volatility_squeeze_threshold"]
        warmup = int(params["warmup"])

        # Calculate momentum
        price = np.nan_to_num(df["close"].values)
        momentum = np.zeros_like(price)
        if momentum_period > 1:
            momentum[1:] = price[1:] - price[:-1]
            momentum = momentum / price[:-1]

        # Extract indicator arrays and handle nans
        ema_long = np.nan_to_num(indicators["ema"][f"ema_{ema_long_period}"])
        ema_short = np.nan_to_num(indicators["ema"][f"ema_{ema_short_period}"])
        atr = np.nan_to_num(indicators["atr"])
        rsi_value = np.nan_to_num(indicators["rsi"][f"rsi_{rsi_period}"])

        # Price relative to EMAs
        above_ema_long = (price > ema_long)
        below_ema_short = (price < ema_short)
        above_ema_short = (price > ema_short)
        below_ema_long = (price < ema_long)

        # Momentum conditions
        momentum_long = momentum > momentum_multiplier
        momentum_short = momentum < -momentum_multiplier

        # RSI conditions
        rsi_long = (rsi_value < rsi_overbought)
        rsi_short = (rsi_value < rsi_oversold)

        # Volatility squeeze - calculate compressed range
        high = np.nan_to_num(df["high"].values)
        low = np.nan_to_num(df["low"].values)
        compressed_range = high - low
        expanded_range = (high - low) / (high + low + 1e-10)
        squeeze = expanded_range < volatility_squeeze_threshold

        # Warmback protection
        signals.iloc[:warmup] = 0.0

        # Long conditions
        long_conditions = (
            above_ema_long &
            below_ema_short &
            momentum_long &
            rsi_long &
            squeeze
        )

        # Short conditions
        short_conditions = (
            below_ema_long &
            above_ema_short &
            momentum_short &
            rsi_short &
            squeeze
        )

        # Apply conditions
        signals[long_conditions] = 1.0
        signals[short_conditions] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v4.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_volatility_lock")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std': 2.5,
         'ema_periods': [21, 50],
         'position_size percent': 2.0,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 4.0,
         'warmup': 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy_v5.py -->
```json
{
  "name": "strategy_v5.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v5.py",
  "ext": ".py",
  "anchor": "strategy_v5_py"
}
```
## strategy_v5_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v5.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_volatility_lock_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr', 'rsi']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'bollinger_period': 20,
         'bollinger_std': 2.5,
         'ema_periods': [21, 50],
         'rsi_period': 14,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 100}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v5.py -->

<!-- MODULE-START: strategy_v6.py -->
```json
{
  "name": "strategy_v6.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v6.py",
  "ext": ".py",
  "anchor": "strategy_v6_py"
}
```
## strategy_v6_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v6.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_patterns")

    @property
    def required_indicators(self) -> List[str]:
        return ['bollinger', 'ema', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14, 'bollinger_periods': [20, 50], 'ema_periods': [12, 21]}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v6.py -->

<!-- MODULE-START: strategy_v7.py -->
```json
{
  "name": "strategy_v7.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v7.py",
  "ext": ".py",
  "anchor": "strategy_v7_py"
}
```
## strategy_v7_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v7.py`  
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
        return ['bollinger', 'ema', 'atr']

    @property
    def default_params(self):
        return {
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'ema_fast': 12,
            'ema_slow': 21,
            'atr_period': 14
        }

    def generate_signals(self, df, indicators, params):
        # Extract parameters
        bollinger_period = params['bollinger_period']
        bollinger_std = params['bollinger_std']
        ema_fast = params['ema_fast']
        ema_slow = params['ema_slow']
        atr_period = params['atr_period']

        # Get indicator arrays
        bollinger = indicators['bollinger']
        ema = indicators['ema']
        atr = indicators['atr']

        # Bollinger Bands
        upper_band = bollinger['upper']
        middle_band = bollinger['middle']
        lower_band = bollinger['lower']

        # EMA values
        ema_fast_arr = ema['ema_{}'.format(ema_fast)]
        ema_slow_arr = ema['ema_{}'.format(ema_slow)]

        # Long conditions
        cond1_long = df['close'] > upper_band
        cond2_long = ema_fast_arr > ema_slow_arr
        cond3_long = self._check_expanding(atr, atr_period)

        long_signals = cond1_long & cond2_long & cond3_long

        # Short conditions
        cond1_short = df['close'] < lower_band
        cond2_short = ema_fast_arr < ema_slow_arr
        cond3_short = self._check_expanding(atr, atr_period)

        short_signals = cond1_short & cond2_short & cond3_short

        # Create signals
        signals = pd.Series(0.0, index=df.index)
        signals[long_signals] = 1.0
        signals[short_signals] = -1.0

        return signals

    def _check_expanding(self, atr_arr: np.ndarray, period: int) -> np.ndarray:
        """
        Check if ATR is expanding (increasing)
        """
        # Calculate ATR differences from previous period
        atr_diff = np.diff(atr_arr, periods=period)

        # Pad with False at the beginning (no previous value for first period)
        is_expanding = np.zeros(len(atr_arr), dtype=bool)
        is_expanding[period:] = (atr_diff > 0)

        return is_expanding
```
<!-- MODULE-END: strategy_v7.py -->

<!-- MODULE-START: strategy_v8.py -->
```json
{
  "name": "strategy_v8.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v8.py",
  "ext": ".py",
  "anchor": "strategy_v8_py"
}
```
## strategy_v8_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v8.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation patterns")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'band_width_threshold': 0.0005,
         'bollinger_period': 20,
         'bollinger_std': 2.0,
         'ema_long': 50,
         'ema_short': 21,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'volume_ratio': 1.1}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v8.py -->

<!-- MODULE-START: strategy_v9.py -->
```json
{
  "name": "strategy_v9.py",
  "path": "20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\\strategy_v9.py",
  "ext": ".py",
  "anchor": "strategy_v9_py"
}
```
## strategy_v9_py
*Path*: `20260211_222053_it_ration_1_erreur_param_tres_hypoth_se\strategy_v9.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="continuation_patterns_v2")

    @property
    def required_indicators(self) -> List[str]:
        return ['ema', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'atr_period': 14,
         'band_width_threshold': 0.0005,
         'bollinger_period': 20,
         'bollinger_std': 2.0,
         'ema_long': 50,
         'ema_short': 21,
         'stop_atr_mult': 2.0,
         'tp_atr_mult': 3.0,
         'warmup_periods': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        close = np.nan_to_num(df['close'].values.astype(np.float64))
        if len(close) < 2:
            return signals
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        long_mask = close > prev_close
        short_mask = close < prev_close
        rsi_raw = indicators.get('rsi')
        if isinstance(rsi_raw, np.ndarray):
            rsi = np.nan_to_num(rsi_raw)
            long_mask = long_mask & (rsi < 55)
            short_mask = short_mask & (rsi > 45)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        signals.iloc[:50] = 0.0
        return signals
```
<!-- MODULE-END: strategy_v9.py -->

<!-- MODULE-START: strategy.py -->
```json
{
  "name": "strategy.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy.py",
  "ext": ".py",
  "anchor": "strategy_py"
}
```
## strategy_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought", type="int", min=50, max=90, default=70
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold", type="int", min=10, max=50, default=30
            ),
            "rsi_period": ParameterSpec(
                name="RSI Period", type="int", min=7, max=28, default=14
            ),
            "stop_atr_mult": ParameterSpec(
                name="Stop ATR Multiplier",
                type="float",
                min=0.5,
                max=3.0,
                default=1.5,
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take Profit ATR Multiplier",
                type="float",
                min=1.0,
                max=6.0,
                default=3.0,
            ),
            "warmup": ParameterSpec(
                name="Warm Up Period", type="int", min=20, max=100, default=50
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        warmup = int(params.get("warmup", 50))

        # Extract indicators
        bb = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bb["upper"])
        lower_bollinger = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators["rsi"])

        # Close prices array
        close = df["close"].values

        # Entry conditions
        long_entry = (close > upper_bollinger) & (rsi < rsi_oversold)
        short_entry = (close < lower_bollinger) & (rsi > rsi_overbought)

        # Exit conditions based on current position
        exit_long = close < lower_bollinger
        exit_short = close > upper_bollinger

        # Apply warmup period
        signals.iloc[:warmup] = 0.0

        # Generate signals
        for i in range(warmup, len(signals)):
            if long_entry[i]:
                signals.iloc[i] = 1.0  # LONG entry
            elif short_entry[i]:
                signals.iloc[i] = -1.0  # SHORT entry
            else:
                if signals.iloc[i-1] == 1.0 and exit_long[i]:
                    signals.iloc[i] = 0.0  # Exit LONG
                elif signals.iloc[i-1] == -1.0 and exit_short[i]:
                    signals.iloc[i] = 0.0  # Exit SHORT

        return signals
```
<!-- MODULE-END: strategy.py -->

<!-- MODULE-START: strategy_v1.py -->
```json
{
  "name": "strategy_v1.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy_v1.py",
  "ext": ".py",
  "anchor": "strategy_v1_py"
}
```
## strategy_v1_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy_v1.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="bollinger_rsi_atr_strategy")

    @property
    def required_indicators(self) -> List[str]:
        return ["bollinger", "rsi"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"bollinger_period": 20, "bollinger_std_dev": 2, "rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 2.0, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # fill each tunable parameter
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        # implement explicit LONG / SHORT / FLAT logic
        # warmup protection
        warmup = int(params.get("warmup", 50))
        signals.iloc[:warmup] = 0.0

        # Access the indicators
        bollinger = indicators["bollinger"]
        rsi = np.nan_to_num(indicators["rsi"])

        # Parameters
        rsi_overbought = params.get("rsi_overbought", 70)
        rsi_oversold = params.get("rsi_oversold", 30)

        # Entry conditions
        long_entry = (np.nan_to_num(df["close"]) > bollinger["upper"]) & (rsi < rsi_oversold)
        short_entry = (np.nan_to_num(df["close"]) < bollinger["lower"]) & (rsi > rsi_overbought)

        # Exit conditions
        exit_condition = (np.nan_to_num(df["close"]) == np.nan_to_num(bollinger["middle"])) | (rsi == 50.0)

        # Apply signals with delays to avoid overlapping
        entries_long = long_entry & ~signals.shift(1).isin([1.0, -1.0])  # Avoid holding both positions
        entries_short = short_entry & ~signals.shift(1).isin([1.0, -1.0])

        # Set signals
        signals[entries_long] = 1.0
        signals[entries_short] = -1.0

        # Exit existing positions when exit condition is met
        current_positions = signals.shift(1)
        exits = (current_positions != 0) & exit_condition
        signals[exits] = 0.0  # Flatten position

        return signals
```
<!-- MODULE-END: strategy_v1.py -->

<!-- MODULE-START: strategy_v2.py -->
```json
{
  "name": "strategy_v2.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy_v2.py",
  "ext": ".py",
  "anchor": "strategy_v2_py"
}
```
## strategy_v2_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy_v2.py`  
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
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self):
        return {}

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index)

        # Extract necessary data as numpy arrays
        close = df['close'].values

        # Get bollinger bands and RSI from indicators
        bollinger = indicators['bollinger']
        upper_band = bollinger['upper']
        lower_band = bollinger['lower']
        rsi = indicators['rsi']

        # Calculate long and short conditions using vectorized operations
        long_mask = (close > upper_band) & (rsi < 30)
        short_mask = (close < lower_band) & (rsi > 70)

        # Assign signals based on conditions
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0

        return signals
```
<!-- MODULE-END: strategy_v2.py -->

<!-- MODULE-START: strategy_v3.py -->
```json
{
  "name": "strategy_v3.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy_v3.py",
  "ext": ".py",
  "anchor": "strategy_v3_py"
}
```
## strategy_v3_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy_v3.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger", "atr"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            # fill each tunable parameter
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        warmup = int(params.get("warmup", 50))

        # Extract necessary data
        close_price = df['close'].values
        rsi_values = np.nan_to_num(indicators["rsi"])
        bb_upper = np.nan_to_num(indicators["bollinger"]["upper"])
        bb_lower = np.nan_to_num(indicators["bollinger"]["lower"])
        atr_values = np.nan_to_num(indicators["atr"])

        # Calculate entry conditions
        long_entry = (close_price > bb_upper) & (rsi_values < params['rsi_oversold'])
        short_entry = (close_price < bb_lower) & (rsi_values > params['rsi_overbought'])

        # Apply warmup period protection
        signals.iloc[:warmup] = 0.0

        # Generate signals
        signals[long_entry] = 1.0  # LONG signal
        signals[short_entry] = -1.0  # SHORT signal

        return signals
```
<!-- MODULE-END: strategy_v3.py -->

<!-- MODULE-START: strategy_v4.py -->
```json
{
  "name": "strategy_v4.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy_v4.py",
  "ext": ".py",
  "anchor": "strategy_v4_py"
}
```
## strategy_v4_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy_v4.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {"rsi_overbought": 70, "rsi_oversold": 30, "rsi_period": 14, "stop_atr_mult": 1.5, "tp_atr_mult": 3.0, "warmup": 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought",
                min=50,
                max=80,
                default=70,
                type=float
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold",
                min=20,
                max=50,
                default=30,
                type=float
            ),
            "rsi_period": ParameterSpec(
                name="RSI Period",
                min=10,
                max=20,
                default=14,
                type=int
            ),
            "stop_atr_mult": ParameterSpec(
                name="Stop ATR Multiplier",
                min=1.0,
                max=3.0,
                default=1.5,
                type=float
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take Profit ATR Multiplier",
                min=2.0,
                max=4.0,
                default=3.0,
                type=float
            ),
            "warmup": ParameterSpec(
                name="Warmup Period",
                min=30,
                max=60,
                default=50,
                type=int
            )
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        # Warmup protection
        warmup = int(params.get("warmup", 50))
        if len(df) <= warmup:
            return signals

        # Extract indicators with proper handling
        rsi = np.nan_to_num(indicators["rsi"])
        bollinger = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bollinger["upper"])
        lower_bollinger = np.nan_to_num(bollinger["lower"])

        # Ensure all arrays are the same length
        min_length = min(len(rsi), len(upper_bollinger), len(lower_bollinger))
        rsi = rsi[:min_length]
        upper_bollinger = upper_bollinger[:min_length]
        lower_bollinger = lower_bollinger[:min_length]

        # Entry conditions
        long_entry = (df['close'].values[:min_length] > upper_bollinger) & (rsi < params["rsi_oversold"])
        short_entry = (df['close'].values[:min_length] < lower_bollinger) & (rsi > params["rsi_overbought"])

        # Exit conditions
        exit_long = df['close'].values[:min_length] < lower_bollinger
        exit_short = df['close'].values[:min_length] > upper_bollinger

        # Apply signals
        signals.iloc[:min_length] = 0.0  # Neutral by default
        signals.iloc[:min_length][long_entry] = 1.0  # LONG signal
        signals.iloc[:min_length][short_entry] = -1.0  # SHORT signal

        # Exit existing positions
        current_position = 0.0
        for i in range(len(signals)):
            if signals[i] == 1.0 or signals[i] == -1.0:
                current_position = signals[i]
            elif exit_long[i] and current_position == 1.0:
                current_position = 0.0
                signals[i] = 0.0
            elif exit_short[i] and current_position == -1.0:
                current_position = 0.0
                signals[i] = 0.0

        return signals
```
<!-- MODULE-END: strategy_v4.py -->

<!-- MODULE-START: strategy_v5.py -->
```json
{
  "name": "strategy_v5.py",
  "path": "20260211_233142_erreur_valueerror_operands_could_not_be\\strategy_v5.py",
  "ext": ".py",
  "anchor": "strategy_v5_py"
}
```
## strategy_v5_py
*Path*: `20260211_233142_erreur_valueerror_operands_could_not_be\strategy_v5.py`  
*Type*: `.py`  

```python
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name="trend_rsi_bollinger")

    @property
    def required_indicators(self) -> List[str]:
        return ["rsi", "bollinger"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "rsi_period": 14,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.0,
            "warmup": 50,
        }

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            "rsi_overbought": ParameterSpec(
                name="RSI Overbought", type="int", min=50, max=90, default=70
            ),
            "rsi_oversold": ParameterSpec(
                name="RSI Oversold", type="int", min=10, max=50, default=30
            ),
            "rsi_period": ParameterSpec(
                name="RSI Period", type="int", min=7, max=28, default=14
            ),
            "stop_atr_mult": ParameterSpec(
                name="Stop ATR Multiplier",
                type="float",
                min=0.5,
                max=3.0,
                default=1.5,
            ),
            "tp_atr_mult": ParameterSpec(
                name="Take Profit ATR Multiplier",
                type="float",
                min=1.0,
                max=6.0,
                default=3.0,
            ),
            "warmup": ParameterSpec(
                name="Warm Up Period", type="int", min=20, max=100, default=50
            ),
        }

    def generate_signals(
        self,
        df: pd.DataFrame,
        indicators: Dict[str, Any],
        params: Dict[str, Any],
    ) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)

        rsi_overbought = params["rsi_overbought"]
        rsi_oversold = params["rsi_oversold"]
        warmup = int(params.get("warmup", 50))

        # Extract indicators
        bb = indicators["bollinger"]
        upper_bollinger = np.nan_to_num(bb["upper"])
        lower_bollinger = np.nan_to_num(bb["lower"])
        rsi = np.nan_to_num(indicators["rsi"])

        # Close prices array
        close = df["close"].values

        # Entry conditions
        long_entry = (close > upper_bollinger) & (rsi < rsi_oversold)
        short_entry = (close < lower_bollinger) & (rsi > rsi_overbought)

        # Exit conditions based on current position
        exit_long = close < lower_bollinger
        exit_short = close > upper_bollinger

        # Apply warmup period
        signals.iloc[:warmup] = 0.0

        # Generate signals
        for i in range(warmup, len(signals)):
            if long_entry[i]:
                signals.iloc[i] = 1.0  # LONG entry
            elif short_entry[i]:
                signals.iloc[i] = -1.0  # SHORT entry
            else:
                if signals.iloc[i-1] == 1.0 and exit_long[i]:
                    signals.iloc[i] = 0.0  # Exit LONG
                elif signals.iloc[i-1] == -1.0 and exit_short[i]:
                    signals.iloc[i] = 0.0  # Exit SHORT

        return signals
```
<!-- MODULE-END: strategy_v5.py -->

## Minimal Directory Tree

`cd D:\backtest_core\sandbox_strategies`

- **20260210_143549_strat_gie_trend_following_avec_bollinger/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
- **20260210_143854_strat_gie_trend_following_avec_bollinger/**
  - strategy.py
  - strategy_v1.py
- **20260210_144605_strat_gie_mean_reversion_btc_30m_avec_rs/**
  - strategy.py
  - strategy_v4.py
- **20260210_160533_cr_e_une_strat_gie_rentable_de_scalping/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
- **20260210_171152_cr_e_une_strat_gie_rentable_de_scalping/**
  - strategy.py
  - strategy_v1.py
- **20260210_174522_cr_e_une_strat_gie_rentable_de_scalping/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v5.py
- **20260211_143609_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
- **20260211_152905_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v2.py
- **20260211_174111_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
- **20260211_174335_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
- **20260211_175901_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
- **20260211_180646_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
- **20260211_180746_mini/**
  - strategy.py
  - strategy_v1.py
- **20260211_181105_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v3.py
  - strategy_v4.py
- **20260211_181856_x/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
- **20260211_181944_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
- **20260211_182712_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
- **20260211_184329_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v2.py
  - strategy_v3.py
- **20260211_192500_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
- **20260211_193252_19_24_49_info_agents_ollama_manager_d_ma/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v5.py
  - strategy_v6.py
- **20260211_195443_warning_data_loader_plus_gros_gap_2019/**
  - strategy.py
  - strategy_v1.py
  - strategy_v10.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v5.py
  - strategy_v6.py
  - strategy_v7.py
  - strategy_v8.py
  - strategy_v9.py
- **20260211_201713_scalp_de_continuation_micro_retournemen/**
  - strategy.py
  - strategy_v1.py
  - strategy_v10.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v7.py
  - strategy_v9.py
- **20260211_222053_it_ration_1_erreur_param_tres_hypoth_se/**
  - strategy.py
  - strategy_v1.py
  - strategy_v10.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v5.py
  - strategy_v6.py
  - strategy_v7.py
  - strategy_v8.py
  - strategy_v9.py
- **20260211_233142_erreur_valueerror_operands_could_not_be/**
  - strategy.py
  - strategy_v1.py
  - strategy_v2.py
  - strategy_v3.py
  - strategy_v4.py
  - strategy_v5.py
