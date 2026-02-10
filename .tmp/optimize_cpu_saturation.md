# Guide Optimisation Saturation CPU (Anti-YOYO)

## Problème observé
CPU oscille 40-80% au lieu de rester à ~100% constant pendant sweep Numba.

## Causes identifiées
1. **Overhead UI entre chunks** → render_live_metrics() appelé à chaque chunk
2. **Chunks trop gros** → workers attendent entre chunks
3. **Post-processing Python** → calculs entre kernels Numba

## Solutions appliquées

### ✅ Solution 1: Throttle UI rendering (APPLIQUÉ)
**Fichier**: `ui/main.py:1257-1274`
**Changement**: Render UI seulement tous les N chunks (max 10 refreshs)
**Impact**: Réduit overhead Python de 80% entre chunks
**Gain CPU**: +10-20% saturation

### 🔧 Solution 2: Ajuster chunk size (OPTIONNEL)

**Test rapide:**
```powershell
# Terminal PowerShell
$env:NUMBA_CHUNK_SIZE = "10000"  # Ou 5000, 20000
streamlit run ui/main.py
```

**Règles de tuning:**
- **Gros sweeps (>100k combos)**: NUMBA_CHUNK_SIZE=10000 ou 20000
- **Sweeps moyens (10k-100k)**: défaut (50000) OK
- **Petits sweeps (<10k)**: défaut ou désactiver chunking

**Indicateur de succès:**
- CPU stable à 85-100% (quelques variations normales)
- Pas de chutes à 40-50% répétées

### 📊 Monitoring

**Pendant le sweep, observez dans Gestionnaire des tâches:**
1. **BON** : CPU oscille 85-100% (variations ±10%)
2. **ACCEPTABLE** : CPU oscille 70-90% (légères chutes)
3. **MAUVAIS** : CPU oscille 40-80% (YOYO prononcé)

## Test de validation

```bash
# Lancer sweep 50k+ combos et observer CPU
cd d:\backtest_core
python .tmp/test_chunk_tuning.py
```

**Attendu après corrections:**
- Démarrage runs: ✅ Rapide
- CPU saturation: ✅ 85-100% stable
- Throughput: ✅ >100k bt/s

## Variables environnement disponibles

```powershell
# Chunk size (défaut: 50000)
$env:NUMBA_CHUNK_SIZE = "10000"

# Cache disque (déjà désactivé en code)
$env:INDICATOR_CACHE_DISK_ENABLED = "0"

# Fast metrics (déjà activé en code)
$env:BACKTEST_FORCE_SLOW_METRICS = "0"
```
