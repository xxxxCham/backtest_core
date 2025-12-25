# backtest_core
Prévision quantitative algorithmique

## Installation
Pré-requis : Windows 11 Pro (24H2), Python 3.11+ et PowerShell. Placez les sources dans `D:\backtest_core` et utilisez un environnement virtuel `.venv` pour isoler les dépendances.

```powershell
Set-Location D:\backtest_core
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if (Test-Path requirements.txt) { pip install -r requirements.txt } else { pip install numpy pandas matplotlib }
```

## Démarrage rapide
1. Activez l’environnement virtuel :

   ```powershell
   Set-Location D:\backtest_core
   .\.venv\Scripts\Activate.ps1
   ```

2. Vérifiez que les dépendances essentielles sont disponibles :

   ```powershell
   python -c "import numpy as np, pandas as pd; print(f'NumPy {np.__version__} / Pandas {pd.__version__}')"
   ```

3. Exécutez un smoke test pour valider l’installation (calcul de variations journalières simulées) :

   ```powershell
   python -c "import pandas as pd; df=pd.DataFrame({'prix':[100,101.5,99.8,102.4,103.1]}); r=df['prix'].pct_change().dropna(); print('Rendements simulés (%) :'); print((r*100).round(2))"
   ```

## Exemple de backtest
L’exemple ci-dessous illustre un backtest vectorisé avec deux moyennes mobiles (courte et longue) générées sur des données synthétiques. Copiez le script dans `backtest_example.py` à la racine du projet, puis exécutez-le depuis PowerShell.

```python
import numpy as np
import pandas as pd


def backtest_sma(data: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
    """
    Calcule un PnL cumulé simple via un croisement de moyennes mobiles.
    - Les signaux sont déclenchés lorsque la moyenne courte croise la longue.
    - Tout est vectorisé avec pandas pour préserver la performance.
    """

    df = data.copy()
    df["sma_short"] = df["close"].rolling(short, min_periods=short).mean()
    df["sma_long"] = df["close"].rolling(long, min_periods=long).mean()
    df["signal"] = np.sign(df["sma_short"] - df["sma_long"])
    df["position"] = df["signal"].ffill().fillna(0)
    df["returns"] = df["close"].pct_change().fillna(0)
    df["strategy"] = df["position"] * df["returns"]
    df["pnl_cumule"] = (1 + df["strategy"]).cumprod() - 1
    return df[["close", "position", "strategy", "pnl_cumule"]]


if __name__ == "__main__":
    rng = pd.date_range("2023-01-02", periods=180, freq="B")
    noise = np.random.default_rng(seed=42).normal(0, 0.004, size=len(rng))
    drift = np.linspace(0, 0.08, num=len(rng))
    price = 100 * (1 + drift + noise).cumprod()
    data = pd.DataFrame({"close": price}, index=rng)

    result = backtest_sma(data)
    stats = {
        "Perf finale (%)": round(result["pnl_cumule"].iloc[-1] * 100, 2),
        "Trades": int(result["position"].diff().fillna(0).abs().sum()),
        "Volatilité (%)": round(result["strategy"].std() * (252 ** 0.5) * 100, 2),
    }

    print("Résumé backtest SMA")
    for k, v in stats.items():
        print(f"- {k}: {v}")
```

Exécution dans PowerShell :

```powershell
Set-Location D:\backtest_core
.\.venv\Scripts\Activate.ps1
python .\backtest_example.py
```
