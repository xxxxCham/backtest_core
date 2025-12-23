# backtest_core
Prévision quantitative algorithmique

See `DETAILS_FONCTIONNEMENT.md` for current operating details and entrypoints.

## Versioned presets
- Naming: `<strategy>@<version>__<preset_slug>`
- Location: `BACKTEST_PRESETS_DIR` or `data/presets`

Example:
```python
from backtest.engine import BacktestEngine
from utils.parameters import save_versioned_preset, load_strategy_version

# after optimization
best_params = {"bb_period": 20, "bb_std": 2.0}

save_versioned_preset(
    strategy_name="bollinger_atr",
    version="0.0.1",
    preset_name="winner",
    params_values=best_params,
)

preset = load_strategy_version("bollinger_atr", version="0.0.1")
params = preset.get_default_values()

engine = BacktestEngine()
# engine.run(df=data, strategy="bollinger_atr", params=params)
```
