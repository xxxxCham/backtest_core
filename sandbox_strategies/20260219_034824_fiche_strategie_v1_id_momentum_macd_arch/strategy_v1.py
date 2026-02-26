from typing import Any, Dict, List

import numpy as np
import pandas as pd

from utils.parameters import ParameterSpec
from strategies.base import StrategyBase


class BuilderGeneratedStrategy(StrategyBase):
    def __init__(self):
        super().__init__(name='momentum_macd')

    @property
    def required_indicators(self) -> List[str]:
        return ['rsi', 'bollinger', 'atr']

    @property
    def default_params(self) -> Dict[str, Any]:
        return {'leverage': 1,
         'rsi_overbought': 70,
         'rsi_oversold': 30,
         'rsi_period': 20,
         'stop_atr_mult': 1.5,
         'tp_atr_mult': 3.0,
         'warmup': 50}

    @property
    def parameter_specs(self) -> Dict[str, ParameterSpec]:
        return {
            'rsi_period': ParameterSpec(
                name='rsi_period',
                min_val=5,
                max_val=50,
                default=20,
                param_type='int',
                step=1,
            ),
            'stop_atr_mult': ParameterSpec(
                name='stop_atr_mult',
                min_val=0.5,
                max_val=4.0,
                default=1.5,
                param_type='float',
                step=0.1,
            ),
            'leverage': ParameterSpec(
                name='leverage',
                min_val=1,
                max_val=2,
                default=1,
                param_type='int',
                step=1,
            ),
            'tp_atr_mult': ParameterSpec(
                name='tp_atr_mult',
                min_val=2.0,
                max_val=4.5,
                default=3.0,
                param_type='float',
                step=0.1,
            ),
        }

    def generate_signals(self, df: pd.DataFrame, indicators: Dict[str, Any], params: Dict[str, Any]) -> pd.Series:
        signals = pd.Series(0.0, index=df.index, dtype=np.float64)
        n = len(df)
        long_mask = np.zeros(n, dtype=bool)
        short_mask = np.zeros(n, dtype=bool)
        # === LOGIQUE LLM INSÉRÉE ICI UNIQUEMENT ===
        class System(object):
            def __init__(self, env_config={}):  # Initialise the system with a configuration dictionary
                self.env = TradingEnvironment(**env_config)  # Initialize the environment
                self._reset()  # Reset the state of the environment

            def _reset(self):
                """Reset all variables to initial values."""
                pass

            @property
            def action_space(self):
                """Return the action space for this system."""
                return self.env.action_space  # This is handled by the TradingEnvironment class

            @property
            def observation_space(self):
                """Return the observation space for this system."""
                return self.env.observation_space  # This is also handled by the TradingEnvironment class

            def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, str, Dict[Any, Any]]:
                """Run an action in the environment."""
                observation, reward, done, info = self.env.step(action)  # Run a single step with the specified action
                return observation, reward, done, "Step completed", {"Action": action}

            def reset(self):
                """Reset the state of the environment."""
                if not self._is_environment_ready():
                    raise RuntimeError("Environment is not ready for use.")  # Raise an error if necessary steps are not taken

                return self.env.reset()  # Reset the environment to its initial state

            def render(self):
                """Render the current observation."""
                pass
        return signals
