# Backtest Core - Référence CLI

> **Ce fichier documente toutes les commandes et fonctionnalités du mode CLI.**  
> À mettre à jour à chaque ajout de commande ou modification de paramètres.

---

## Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Commandes disponibles](#commandes-disponibles)
3. [Paramètres globaux](#paramètres-globaux)
4. [Exemples d'utilisation](#exemples-dutilisation)
5. [Historique des fonctions](#historique-des-fonctions)

---

## Vue d'ensemble

Le mode CLI permet d'exécuter des backtests, optimisations et analyses sans passer par l'interface Streamlit. Idéal pour :
- Automatisation via scripts
- Exécution en batch
- Intégration CI/CD
- Contrôle programmatique par agents LLM

**Point d'entrée principal** : `python -m backtest_core` (à implémenter)

---

## Commandes disponibles

### `backtest` - Exécuter un backtest simple

```bash
python -m backtest_core backtest [OPTIONS]
```

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `--strategy` | str | requis | Nom de la stratégie (`bollinger_atr`, `ema_cross`, etc.) |
| `--data` | path | requis | Chemin vers fichier OHLCV (Parquet, CSV, JSON) |
| `--params` | json | `{}` | Paramètres stratégie en JSON |
| `--capital` | float | `10000` | Capital initial |
| `--fees-bps` | int | `10` | Frais en basis points |
| `--output` | path | `None` | Fichier de sortie résultats |
| `--format` | str | `json` | Format sortie: `json`, `csv`, `parquet` |

**Exemple :**
```powershell
$env:BACKTEST_DATA_DIR = "D:\path\to\parquet"
python __main__.py backtest -s ema_cross -d BTCUSDC_1h.parquet
python __main__.py backtest -s bollinger_dual -d ETHUSDC_15m.parquet --capital 50000
```

**Status** : ✅ Implémenté (12/12/2025)

---

### `sweep` - Optimisation paramétrique

```bash
python -m backtest_core sweep [OPTIONS]
```

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `--strategy` | str | requis | Nom de la stratégie |
| `--data` | path | requis | Chemin données OHLCV |
| `--granularity` | float | `0.5` | Granularité (0.0=fin, 1.0=grossier) |
| `--max-combinations` | int | `10000` | Limite combinaisons |
| `--metric` | str | `sharpe` | Métrique d'optimisation |
| `--parallel` | int | `4` | Nombre de workers |
| `--output` | path | `None` | Fichier résultats |

**Métriques disponibles :**
- `sharpe` - Sharpe Ratio
- `sortino` - Sortino Ratio
- `total_return` - Rendement total
- `max_drawdown` - Drawdown maximum
- `win_rate` - Taux de trades gagnants
- `profit_factor` - Facteur de profit

**Exemple :**
```powershell
python __main__.py sweep -s ema_cross -d BTCUSDC_4h.parquet --granularity 0.3 --metric sharpe
python __main__.py sweep -s ema_cross -d BTCUSDC_1h.parquet --granularity 0.9 --top 5 -o sweep_result.json
```

**Status** : ✅ Implémenté (13/12/2025)

---

### `list` - Lister les ressources

```bash
python -m backtest_core list [RESOURCE]
```

| Resource | Description |
|----------|-------------|
| `strategies` | Liste des stratégies enregistrées |
| `indicators` | Liste des indicateurs disponibles |
| `data` | Fichiers de données découverts |
| `presets` | Presets de configuration |

**Exemple :**
```powershell
$env:BACKTEST_DATA_DIR = "D:\path\to\parquet"
python __main__.py list strategies
python __main__.py list indicators  
python __main__.py list data
```

**Status** : ✅ Implémenté (12/12/2025)

---

### `info` - Informations détaillées

```bash
python -m backtest_core info [RESOURCE] [NAME]
```

Affiche les paramètres, plages d'optimisation et documentation d'une stratégie ou indicateur.

**Exemple :**
```powershell
python __main__.py info strategy bollinger_dual
python __main__.py info indicator supertrend
```

**Status** : ✅ Implémenté (12/12/2025)

---

### `validate` - Valider configuration

```bash
python -m backtest_core validate [OPTIONS]
```

Vérifie l'intégrité des stratégies, indicateurs et données.

| Paramètre | Description |
|-----------|-------------|
| `--strategy NAME` | Valider une stratégie spécifique |
| `--data PATH` | Valider un fichier de données |
| `--all` | Valider tout le système |

**Exemple :**
```powershell
python __main__.py validate --all
```

**Status** : ✅ Implémenté (12/12/2025)

---

### `export` - Exporter résultats

```bash
python -m backtest_core export [OPTIONS]
```

| Paramètre | Description |
|-----------|-------------|
| `-i, --input` | Fichier résultats à exporter (JSON) |
| `-f, --format` | Format: `html`, `csv`, `excel` (défaut: html) |
| `-o, --output` | Fichier de sortie |
| `--template` | Template de rapport personnalisé |

**Formats supportés :**
- `html` - Rapport HTML avec métriques stylées
- `csv` - Export CSV pour analyse externe
- `excel` - Export Excel (requiert openpyxl)

**Exemple :**
```powershell
python __main__.py export -i sweep_result.json -f html -o report.html
python __main__.py export -i sweep_result.json -f csv -o results.csv
```

**Status** : ✅ Implémenté (13/12/2025)

---

### `optuna` - Optimisation bayésienne

```bash
python -m backtest_core optuna [OPTIONS]
```

Optimisation intelligente des paramètres via Optuna. **10-100x plus rapide** que le sweep classique car utilise l'algorithme TPE (Tree-structured Parzen Estimator) au lieu d'un grid search exhaustif.

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `-s, --strategy` | str | requis | Nom de la stratégie |
| `-d, --data` | path | requis | Chemin données OHLCV |
| `-n, --n-trials` | int | `100` | Nombre de trials (itérations) |
| `-m, --metric` | str | `sharpe` | Métrique à optimiser (ou multi: `sharpe,max_drawdown`) |
| `--sampler` | str | `tpe` | Algorithme: `tpe`, `cmaes`, `random` |
| `--pruning` | flag | `false` | Activer le pruning (arrêt précoce) |
| `--pruner` | str | `median` | Type de pruner: `median`, `hyperband` |
| `--multi-objective` | flag | `false` | Mode multi-objectif (Pareto) |
| `--param-space` | json | auto | Espace paramètres personnalisé |
| `-c, --constraints` | list | `[]` | Contraintes (ex: `slow_period,>,fast_period`) |
| `--timeout` | int | `None` | Timeout en secondes |
| `--parallel` | int | `1` | Jobs parallèles |
| `--capital` | float | `10000` | Capital initial |
| `--fees-bps` | int | `10` | Frais en basis points |
| `--top` | int | `10` | Nombre de résultats à afficher |
| `-o, --output` | path | `None` | Fichier de sortie |
| `--early-stop-patience` | int | `None` | **[17/12/2025]** Arrêt anticipé après N trials sans amélioration |

**Avantages vs Sweep :**
- 🚀 **100 trials Optuna ≈ 10000 combinaisons sweep** en qualité
- 🧠 Algorithme bayésien (TPE) explore intelligemment
- ✂️ Pruning stoppe les runs peu prometteurs
- 📊 Support multi-objectif (frontière Pareto)

**Exemples :**
```powershell
# Optimisation simple (100 trials)
python __main__.py optuna -s ema_cross -d BTCUSDC_1h.parquet

# Avec pruning et plus de trials
python __main__.py optuna -s bollinger_atr -d data.parquet -n 200 --pruning

# Avec early stopping (arrêt après 15 trials sans amélioration)
python __main__.py optuna -s ema_cross -d data.parquet -n 200 --early-stop-patience 15

# Avec contraintes
python __main__.py optuna -s ema_cross -d data.parquet -c slow_period,>,fast_period

# Multi-objectif (Pareto: maximiser Sharpe, minimiser drawdown)
python __main__.py optuna -s ema_cross -d data.parquet -m "sharpe,max_drawdown" --multi-objective

# Combinaison pruning + early stopping
python __main__.py optuna -s bollinger_atr -d data.parquet --pruning --early-stop-patience 10

# Export résultats
python __main__.py optuna -s ema_cross -d data.parquet -o optuna_results.json
```

**Usage Python :**
```python
from backtest import OptunaOptimizer, quick_optimize

# Quick optimize
result = quick_optimize("ema_cross", df, n_trials=100)
print(result.best_params)  # {'fast_period': 12, 'slow_period': 45}

# Avec contraintes
result = quick_optimize(
    "ema_cross", df,
    param_space={
        "fast_period": {"type": "int", "low": 5, "high": 50},
        "slow_period": {"type": "int", "low": 20, "high": 200},
    },
    constraints=[("slow_period", ">", "fast_period")],
)
```

**Status** : ✅ Implémenté (16/12/2025)

---

### `visualize` - Visualisation interactive

```bash
python -m backtest_core visualize [OPTIONS]
```

Génère des graphiques interactifs avec Plotly : candlesticks OHLCV, marqueurs de trades (entrées/sorties), et rapport HTML complet.

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `-i, --input` | path | requis | Fichier JSON de résultats (backtest, sweep ou optuna) |
| `-d, --data` | path | optionnel | Fichier OHLCV pour les candlesticks |
| `-o, --output` | path | auto | Fichier HTML de sortie |
| `--html` | flag | `false` | Générer automatiquement un HTML |
| `-m, --metric` | str | `sharpe_ratio` | Métrique pour sélectionner le meilleur (sweep/optuna) |
| `--capital` | float | `10000` | Capital initial |
| `--fees-bps` | int | `10` | Frais en basis points |
| `--no-show` | flag | `false` | Ne pas ouvrir le graphique dans le navigateur |

**Fonctionnalités :**
- 📊 **Candlesticks** : Bougies OHLCV avec les données réelles
- 🎯 **Marqueurs de trades** : 
  - Triangles verts (▲) : Entrées LONG
  - Triangles rouges (▼) : Entrées SHORT
  - Cercles verts/rouges : Sorties (win/loss)
- 💡 **Tooltips interactifs** : PnL, prix, raison de sortie
- 📈 **Equity curve** : Courbe de capital + high water mark
- 📉 **Drawdown** : Graphique des pertes maximales
- 📋 **Table des trades** : Détail de tous les trades

**Exemples :**
```powershell
# Visualiser un backtest avec données OHLCV
python __main__.py visualize -i results.json -d data.csv --html

# Visualiser les résultats d'un sweep (prend le meilleur)
python __main__.py visualize -i sweep_results.json -d data.parquet

# Sans ouvrir le navigateur (juste générer HTML)
python __main__.py visualize -i results.json -d data.csv --html --no-show

# Utiliser une métrique spécifique pour le tri
python __main__.py visualize -i sweep.json -d data.csv -m sortino_ratio
```

**Status** : ✅ Implémenté (17/12/2025)

---

## Paramètres globaux

Ces paramètres s'appliquent à toutes les commandes :

| Paramètre | Description |
|-----------|-------------|
| `--verbose`, `-v` | Mode verbose (debug) |
| `--quiet`, `-q` | Mode silencieux |
| `--config FILE` | Fichier de configuration TOML |
| `--seed INT` | Seed pour reproductibilité |
| `--no-color` | Désactiver couleurs terminal |

---

## Exemples d'utilisation

### Pipeline complet d'optimisation

```bash
# 1. Lister les stratégies disponibles
python -m backtest_core list strategies

# 2. Voir les paramètres d'une stratégie
python -m backtest_core info strategy bollinger_dual

# 3. Lancer l'optimisation
python -m backtest_core sweep \
    --strategy bollinger_dual \
    --data data/BTCUSDT_1h.parquet \
    --granularity 0.4 \
    --metric sharpe \
    --output results/bollinger_dual_sweep.json

# 4. Backtest avec les meilleurs paramètres
python -m backtest_core backtest \
    --strategy bollinger_dual \
    --data data/BTCUSDT_1h.parquet \
    --params '{"bb_window": 25, "bb_std": 2.2, "ma_window": 12}' \
    --output results/bollinger_dual_best.json
```

### Batch sur plusieurs stratégies

```bash
for strategy in bollinger_atr ema_cross macd_cross; do
    python -m backtest_core sweep \
        --strategy $strategy \
        --data data/BTCUSDT_4h.parquet \
        --output results/${strategy}_sweep.json
done
```

### Utilisation avec fichier config

```toml
# config/my_sweep.toml
[sweep]
strategy = "bollinger_dual"
data = "data/BTCUSDT_1h.parquet"
granularity = 0.3
metric = "sharpe"
parallel = 8

[params]
bb_window = [15, 20, 25, 30]
bb_std = [1.8, 2.0, 2.2]
ma_window = [8, 10, 12, 15]
```

```bash
python -m backtest_core sweep --config config/my_sweep.toml
```

---

## Historique des fonctions

> Chaque nouvelle commande ou modification doit être documentée ici.

| Date | Commande | Action | Description |
|------|----------|--------|-------------|
| 12/12/2025 | - | Création | Création du fichier CLI_REFERENCE.md |
| 12/12/2025 | `backtest` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `sweep` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `list` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `info` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `validate` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `export` | Spécification | Définition de la commande (non implémentée) |
| 12/12/2025 | `list` | Implémentation | Commande list fonctionnelle (strategies, indicators, data) |
| 12/12/2025 | `info` | Implémentation | Commande info fonctionnelle |
| 12/12/2025 | `backtest` | Implémentation | Commande backtest fonctionnelle avec données réelles |
| 12/12/2025 | `validate` | Implémentation | Commande validate fonctionnelle |
| 13/12/2025 | `sweep` | Implémentation | Commande sweep fonctionnelle avec grille paramétrique |
| 13/12/2025 | `export` | Implémentation | Commande export fonctionnelle (HTML, CSV, Excel) |
| 16/12/2025 | `optuna` | Implémentation | Optimisation bayésienne via Optuna (TPE, CMA-ES, pruning, multi-objectif) |
| 17/12/2025 | `visualize` | Implémentation | Visualisation interactive (candlesticks + trades + rapport HTML) |
| 17/12/2025 | `optuna` | Amélioration | Ajout argument `--early-stop-patience` pour arrêt anticipé |

---

## Notes pour les agents LLM

> **Directive** : Lors de l'implémentation d'une commande CLI :
> 1. Mettre à jour le status de 🔜 vers ✅
> 2. Ajouter une entrée dans l'historique avec la date
> 3. Documenter tout nouveau paramètre ajouté
> 4. Mentionner dans `copilot-instructions.md` → Index des Modifications

---

*Dernière mise à jour : 16/12/2025*
