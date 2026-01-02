# Système d'Arrêt d'Urgence - Guide d'Utilisation

## 📋 Vue d'ensemble

Le système d'arrêt d'urgence permet d'arrêter complètement un backtest en cours et de libérer toute la mémoire (RAM + VRAM) pour pouvoir relancer immédiatement un nouveau test sans redémarrer l'application.

## ✨ Fonctionnalités

### Nettoyage complet en 9 étapes

1. **Arrêt des opérations** - Signale l'arrêt aux sweep engines et agents LLM
2. **Déchargement LLM** - Décharge tous les modèles Ollama de la VRAM
3. **Cache indicateurs** - Nettoie les indicateurs expirés et le cache mémoire
4. **CuPy GPU** - Libère tous les memory pools GPU (VRAM)
5. **PyTorch CUDA** - Vide le cache PyTorch si présent
6. **MemoryManager** - Nettoie tous les caches managés
7. **Garbage Collector** - 3 passes agressives pour libérer la RAM
8. **Session State** - Réinitialise les flags Streamlit
9. **Mesure mémoire** - Calcule la mémoire libérée

## 🎯 Utilisation dans l'UI Streamlit

### Bouton "Arrêt d'urgence"

1. Lancer un backtest (mode simple, grille ou LLM)
2. Pendant l'exécution, cliquer sur **"⛔ Arrêt d'urgence"**
3. Le système effectue le nettoyage complet (1-2 secondes)
4. Un message de confirmation s'affiche avec le nombre de composants nettoyés
5. L'application est prête pour un nouveau test immédiatement

### Feedback visuel

```
✅ Arrêt réussi : 5 composants nettoyés
💡 Système prêt pour un nouveau test

📊 Détails du nettoyage (cliquer pour développer)
```

Si des erreurs surviennent (ex: module non installé), elles sont affichées mais n'empêchent pas le nettoyage des autres composants.

## 🔧 Utilisation programmatique

### Exemple simple

```python
from ui.emergency_stop import execute_emergency_stop

# Dans un script ou fonction
stats = execute_emergency_stop(st.session_state)

print(f"✅ {len(stats['components_cleaned'])} composants nettoyés")
print(f"❌ {len(stats['errors'])} erreurs")
```

### Exemple avancé avec gestionnaire

```python
from ui.emergency_stop import get_emergency_handler

handler = get_emergency_handler()

# Demander l'arrêt
handler.request_stop()

# Vérifier si arrêt demandé
if handler.is_stop_requested():
    # Effectuer le nettoyage
    stats = handler.full_cleanup(session_state=st.session_state)

    # Analyser les résultats
    print(f"RAM libérée: {stats.get('ram_freed_mb', 0):.2f} MB")
    print(f"VRAM libérée: {stats.get('vram_freed_mb', 0):.2f} MB")

# Réinitialiser le flag
handler.reset_stop()
```

## 📊 Structure des statistiques retournées

```python
{
    "timestamp": 1767285356.8190339,
    "components_cleaned": [
        "session_flags",
        "sweep_engine_signal",
        "indicator_memory_cache",
        "cupy_memory_pool",
        "cupy_pinned_pool",
        "garbage_collector"
    ],
    "errors": [],  # Liste des erreurs rencontrées
    "ram_freed_mb": 0.0,
    "vram_freed_mb": 0.0,
    "current_ram_mb": 1234.56,  # Usage RAM actuel (si psutil disponible)
    "gc_collected_objects": 42  # Objets collectés par le GC
}
```

## 🧪 Tests

### Tester le système

```bash
python test_emergency_stop.py
```

Sortie attendue :
```
============================================================
TEST DU SYSTÈME D'ARRÊT D'URGENCE
============================================================

1️⃣ Création du gestionnaire...
   ✅ Handler créé avec succès

2️⃣ Test flag d'arrêt...
   ✅ Flags fonctionnent correctement

3️⃣ Test nettoyage complet...
   ⏱️  Durée: 1.65s
   🧹 Composants nettoyés: 5
   ❌ Erreurs: 0

============================================================
✅ TOUS LES TESTS PASSÉS
============================================================
```

## ⚠️ Notes importantes

### Limitations

1. **Processus parallèles** : Les backtests déjà lancés en multiprocess continueront jusqu'à leur fin, mais aucun nouveau ne sera démarré
2. **Threads** : Impossible de tuer brutalement les threads Python sans risque de corruption
3. **Cache indicateurs** : Par défaut conservé sur disque (seul le cache mémoire est vidé)

### Erreurs normales

Certaines erreurs sont normales si les modules ne sont pas installés :
- `pytorch: No module named 'torch'` - PyTorch non installé
- `cupy: No module named 'cupy'` - CuPy non installé

Ces erreurs n'empêchent pas le nettoyage des autres composants.

## 🔍 Architecture technique

### Classes principales

- **`EmergencyStopHandler`** : Gestionnaire principal (singleton)
- **`get_emergency_handler()`** : Obtenir l'instance singleton
- **`execute_emergency_stop()`** : Raccourci pour nettoyage complet

### Composants nettoyés

| Composant | Méthode | Impact |
|-----------|---------|--------|
| Session flags | `_stop_running_operations()` | Arrête les boucles en cours |
| LLM Ollama | `_cleanup_llm_models()` | Libère VRAM (unload via API) |
| Cache indicateurs | `_cleanup_indicator_cache()` | Libère RAM |
| CuPy pools | `_cleanup_cupy()` | Libère VRAM GPU |
| PyTorch CUDA | `_cleanup_pytorch()` | Libère VRAM GPU |
| MemoryManager | `_cleanup_memory_manager()` | Libère RAM |
| Garbage Collector | `_aggressive_gc()` | Libère RAM (3 passes) |
| Session state | `_reset_session_state()` | Réinitialise flags UI |

### Intégration avec SweepEngine

Le `SweepEngine` vérifie automatiquement le flag `_stop_requested` dans sa boucle de traitement (ligne 332-334) :

```python
if self._stop_requested:
    logger.warning("🛑 Arrêt d'urgence détecté - Interruption du sweep")
    break
```

## 🚀 Améliorations futures

### Optionnelles

1. **Nettoyage complet cache** : Ajouter option pour `bank.clear()` (vide cache disque)
2. **Kill brutal processus** : Implémenter terminaison forcée des processus multiprocess (risque corruption)
3. **Monitoring temps réel** : Afficher progression du nettoyage étape par étape
4. **Auto-save avant arrêt** : Sauvegarder résultats partiels avant arrêt

## 📝 Changelog

- **01/01/2026** : Création du système d'arrêt d'urgence complet
  - 10 composants nettoyés
  - Gestion d'erreurs granulaire
  - Tests automatisés
  - Documentation complète
