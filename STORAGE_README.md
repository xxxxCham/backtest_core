# 📦 Système de Stockage des Résultats de Backtests

## Vue d'ensemble

Le système de stockage permet de **sauvegarder et charger automatiquement** les résultats des backtests et sweeps dans un format structuré et performant.

### Fonctionnalités principales

✅ **Sauvegarde automatique** des résultats de backtests
✅ **Format hybride** : JSON (métadonnées) + Parquet (séries temporelles)
✅ **Index searchable** pour recherche rapide
✅ **Compression optionnelle** pour économiser l'espace disque
✅ **Gestion automatique** : nettoyage des anciens résultats
✅ **Support des sweeps** : sauvegarde de grilles d'optimisation complètes

---

## 🚀 Utilisation rapide

### 1. Sauvegarde automatique (activée par défaut)

```python
from backtest.engine import BacktestEngine
from strategies.bollinger_atr import BollingerATRStrategy

# auto_save=True par défaut
engine = BacktestEngine(initial_capital=10000)

result = engine.run(
    df=data,
    strategy=BollingerATRStrategy(),
    params={"entry_z": 2.0},
    symbol="BTCUSDT",
    timeframe="1h"
)

# ✅ Le résultat est automatiquement sauvegardé dans backtest_results/
print(f"Résultat sauvegardé: {result.meta['run_id']}")
```

### 2. Charger un résultat

```python
from backtest.storage import get_storage

storage = get_storage()

# Charger par run_id
result = storage.load_result("run_20231215_143022")

print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Total Trades: {len(result.trades)}")
```

### 3. Rechercher des résultats

```python
# Lister tous les résultats (triés par date)
all_results = storage.list_results(limit=10)

# Filtrer par stratégie
bollinger_runs = storage.search_results(strategy="bollinger_atr")

# Filtrer par performance
good_runs = storage.search_results(
    min_sharpe=1.5,
    max_drawdown=20.0,
    min_trades=10
)

# Obtenir les meilleurs résultats
top_10 = storage.get_best_results(n=10, metric="sharpe_ratio")

for meta in top_10:
    print(f"{meta.run_id}: Sharpe={meta.metrics['sharpe_ratio']:.2f}")
```

### 4. Sauvegarder des sweeps

```python
from backtest.sweep import SweepEngine

# auto_save=True par défaut
engine = SweepEngine(max_workers=8)

sweep_results = engine.run_sweep(
    df=data,
    strategy="bollinger_atr",
    param_grid={
        "entry_z": [1.5, 2.0, 2.5],
        "k_sl": [1.0, 1.5, 2.0]
    }
)

# ✅ Le sweep complet est automatiquement sauvegardé

# Charger un sweep
sweep_data = storage.load_sweep_results("sweep_20231215_150000")
print(sweep_data["summary"])
print(sweep_data["results_df"].head())
```

---

## 📁 Structure de stockage

```
backtest_results/
├── index.json                  # Catalogue de tous les runs
├── run_20231215_143022/
│   ├── metadata.json          # Paramètres, métriques, config
│   ├── equity.parquet         # Courbe d'équité (Series)
│   ├── returns.parquet        # Rendements (Series)
│   └── trades.parquet         # Historique des trades (DataFrame)
├── run_20231215_143500/
│   └── ...
├── sweep_20231215_150000/
│   ├── summary.json           # Résumé du sweep
│   └── all_results.parquet    # Tous les résultats de la grille
└── ...
```

### Format des fichiers

- **JSON** : Métadonnées lisibles par l'humain
- **Parquet** : Séries temporelles compressées et rapides à charger
- **Index** : Permet la recherche sans charger tous les fichiers

---

## 🔧 Configuration avancée

### Désactiver la sauvegarde automatique

```python
# Pour un backtest unique
engine = BacktestEngine(auto_save=False)

# Pour un sweep
sweep_engine = SweepEngine(auto_save=False)
```

### Utiliser un répertoire personnalisé

```python
from backtest.storage import ResultStorage

storage = ResultStorage(
    storage_dir="/path/to/custom/storage",
    auto_save=True,
    compress=True  # Activer la compression Parquet
)

# Sauvegarder manuellement
storage.save_result(result)
```

### Singleton global

```python
from backtest.storage import get_storage

# Retourne toujours la même instance
storage = get_storage()
```

---

## 🔍 API complète

### ResultStorage

#### Sauvegarde

```python
# Sauvegarder un backtest
run_id = storage.save_result(
    result: RunResult,
    run_id: Optional[str] = None,  # Auto-généré si None
    auto_cleanup: bool = False      # Nettoyer anciens résultats
) -> str

# Sauvegarder un sweep
sweep_id = storage.save_sweep_results(
    sweep_results: SweepResults,
    sweep_id: Optional[str] = None
) -> str
```

#### Chargement

```python
# Charger un backtest
result = storage.load_result(run_id: str) -> RunResult

# Charger un sweep
sweep_data = storage.load_sweep_results(sweep_id: str) -> Dict
# Retourne: {"summary": dict, "results_df": DataFrame, "sweep_id": str}
```

#### Recherche

```python
# Lister tous les résultats
results = storage.list_results(
    limit: Optional[int] = None,
    sort_by: str = "timestamp",      # ou "sharpe_ratio", "total_return"
    reverse: bool = True             # Tri descendant
) -> List[StoredResultMetadata]

# Recherche avec filtres
results = storage.search_results(
    strategy: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    min_sharpe: Optional[float] = None,
    max_drawdown: Optional[float] = None,
    min_trades: Optional[int] = None,
    date_from: Optional[str] = None,  # ISO format
    date_to: Optional[str] = None
) -> List[StoredResultMetadata]

# Meilleurs résultats
best = storage.get_best_results(
    n: int = 10,
    metric: str = "sharpe_ratio"     # ou autre métrique
) -> List[StoredResultMetadata]
```

#### Gestion

```python
# Supprimer un résultat
success = storage.delete_result(run_id: str) -> bool

# Nettoyer anciens résultats
deleted = storage._cleanup_old_results(keep_last: int = 1000) -> int

# Tout supprimer (⚠️ DANGER)
success = storage.clear_all() -> bool

# Reconstruire l'index (en cas de corruption)
count = storage.rebuild_index() -> int
```

---

## 📊 StoredResultMetadata

Métadonnées d'un résultat sauvegardé :

```python
@dataclass
class StoredResultMetadata:
    run_id: str              # Identifiant unique
    timestamp: str           # Date de création (ISO)
    strategy: str            # Nom de la stratégie
    symbol: str              # Symbole tradé
    timeframe: str           # Timeframe des données
    params: Dict[str, Any]   # Paramètres utilisés
    metrics: Dict[str, Any]  # Métriques de performance
    n_bars: int              # Nombre de barres
    n_trades: int            # Nombre de trades
    period_start: str        # Début de période
    period_end: str          # Fin de période
    duration_sec: float      # Durée d'exécution
```

---

## 🎯 Exemples d'utilisation

### Comparer plusieurs stratégies

```python
from backtest.storage import get_storage

storage = get_storage()

# Récupérer tous les résultats pour BTCUSDT
results = storage.search_results(symbol="BTCUSDT")

# Grouper par stratégie
strategies = {}
for meta in results:
    strat = meta.strategy
    if strat not in strategies:
        strategies[strat] = []
    strategies[strat].append(meta.metrics["sharpe_ratio"])

# Afficher les moyennes
for strat, sharpes in strategies.items():
    avg_sharpe = sum(sharpes) / len(sharpes)
    print(f"{strat}: Sharpe moyen = {avg_sharpe:.2f} ({len(sharpes)} runs)")
```

### Trouver le meilleur run par période

```python
# Meilleurs runs du mois dernier
from datetime import datetime, timedelta

date_from = (datetime.now() - timedelta(days=30)).isoformat()

recent_best = storage.search_results(
    date_from=date_from,
    min_sharpe=1.0
)

# Charger le meilleur
if recent_best:
    best_meta = max(recent_best, key=lambda x: x.metrics["sharpe_ratio"])
    result = storage.load_result(best_meta.run_id)
    print(f"Meilleur run: {result.summary()}")
```

### Export vers CSV/Excel

```python
# Charger un résultat
result = storage.load_result("run_20231215_143022")

# Exporter les trades
result.trades.to_csv("trades.csv", index=False)

# Exporter l'équité
result.equity.to_csv("equity.csv")

# Exporter vers Excel
with pd.ExcelWriter("backtest_results.xlsx") as writer:
    result.trades.to_excel(writer, sheet_name="Trades")
    result.equity.to_frame().to_excel(writer, sheet_name="Equity")
```

---

## ⚙️ Performance

### Tailles de fichiers (typique)

- **metadata.json** : ~2-5 KB
- **equity.parquet** : ~50-200 KB (1000 points)
- **trades.parquet** : ~10-50 KB (100 trades)
- **Total par run** : ~100-500 KB

### Vitesse

- **Sauvegarde** : ~50-100ms par run
- **Chargement** : ~20-50ms par run
- **Recherche index** : <1ms (sans chargement)

### Recommandations

- Activer `compress=True` pour économiser 30-50% d'espace
- Nettoyer régulièrement avec `_cleanup_old_results()`
- Limiter à ~1000 runs pour maintenir les performances
- Utiliser `list_results(limit=N)` pour éviter de charger tout l'index

---

## 🧪 Tests

Exécuter les tests du système de stockage :

```bash
# Tous les tests de storage
pytest tests/test_storage.py -v

# Tests spécifiques
pytest tests/test_storage.py::test_save_result -v
pytest tests/test_storage.py::test_search_results -v
```

---

## 🎬 Démonstration

Exécuter le script de démonstration complet :

```bash
python demo/demo_storage.py
```

Ce script montre :
1. Sauvegarde et chargement basiques
2. Recherche et filtrage
3. Stockage des sweeps
4. Gestion des résultats
5. Chargement et analyse

---

## 🔒 Sécurité et fiabilité

### Gestion des erreurs

- Les erreurs de sauvegarde n'interrompent **pas** le backtest
- En cas d'échec, un warning est loggé mais l'exécution continue
- Les fichiers partiellement écrits sont nettoyés automatiquement

### Intégrité des données

- Validation des données avant sauvegarde
- Index auto-réparable avec `rebuild_index()`
- Format Parquet garantit l'intégrité des séries temporelles

### Compatibilité

- Compatible Windows, Linux, macOS
- Pas de dépendances sur la structure du code
- Migration facile (copier le dossier `backtest_results/`)

---

## 📝 Notes importantes

1. **Emplacement par défaut** : `./backtest_results/` (relatif au CWD)
2. **Auto-save activé** : Par défaut pour `BacktestEngine` et `SweepEngine`
3. **Run ID unique** : Généré automatiquement (format: `run_YYYYMMDD_HHMMSS`)
4. **Persistence** : L'index est sauvegardé à chaque modification
5. **Singleton** : `get_storage()` retourne toujours la même instance

---

## 🚀 Intégration avec l'UI

Le système de stockage est intégré avec `BackendFacade` pour l'UI :

```python
from backtest.facade import BackendFacade, BacktestRequest
from backtest.storage import get_storage

# Exécuter via la façade (sauvegarde automatique)
facade = BackendFacade()
response = facade.run_backtest(request)

# Récupérer l'historique pour l'UI
storage = get_storage()
history = storage.list_results(limit=20)

# Afficher dans l'UI
for meta in history:
    print(f"{meta.timestamp}: {meta.strategy} - Sharpe={meta.metrics['sharpe_ratio']:.2f}")
```

---

## 🔗 Ressources

- **Code source** : [backtest/storage.py](backtest/storage.py)
- **Tests** : [tests/test_storage.py](tests/test_storage.py)
- **Démo** : [demo/demo_storage.py](demo/demo_storage.py)
- **Engine** : [backtest/engine.py](backtest/engine.py)
- **Sweep** : [backtest/sweep.py](backtest/sweep.py)

---

**Auteur** : backtest_core
**Version** : 1.0.0
**Date** : Décembre 2025
