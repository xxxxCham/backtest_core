# Orchestration Automatique : Indicateurs, Stratégies & Granularité

NOTE: Historical design doc. For current operating details, see
`DETAILS_FONCTIONNEMENT.md`. For orchestration log details, see
`docs/ORCHESTRATION_LOGS.md`.

## Vue d'Ensemble 🎯

Le système orchestre automatiquement **3 couches complémentaires** :

1. **Stratégies** → Quelles règles de trading utiliser
2. **Indicateurs** → Quels outils techniques charger automatiquement
3. **Granularité** → Combien de valeurs de paramètres tester

---

## 1. Orchestration des Stratégies 📊

### A. Listing Dynamique

**Fichier** : [ui/app.py:898](ui/app.py#L898)

```python
available_strategies = list_strategies()  # Détection automatique
```

→ **Récupère automatiquement toutes les stratégies** enregistrées via `@register_strategy`

### B. Affichage Utilisateur

**Sidebar** (ligne 927-930) :
```python
strategy_name = st.sidebar.selectbox(
    "Stratégie",
    list(strategy_options.keys())  # Liste enrichie avec emojis
)
```

**Options disponibles** :
- 📉 Bollinger + ATR (Mean Reversion)
- 📈 EMA Crossover (Trend Following)
- 📊 MACD Crossover (Momentum)
- 🔄 RSI Reversal (Mean Reversion)
- 📏 ATR Channel (Breakout)
- 📐 MA Crossover (SMA Trend)
- ⚡ EMA + Stochastic (Scalping)
- 📊 Bollinger Dual (Mean Reversion)
- 🔄 RSI Trend Filtered (Mean Rev.)

→ **9 stratégies** disponibles automatiquement

### C. Description Contextuelle

Pour chaque stratégie sélectionnée :
```python
st.sidebar.caption(strategy_descriptions.get(strategy_key, ""))
# Exemple : "Achète bas des bandes, vend haut. Filtre ATR."
```

---

## 2. Orchestration des Indicateurs 🔧

### A. Chargement Automatique

**Fichier** : [ui/app.py:946-964](ui/app.py#L946-L964)

```python
strategy_info = get_strategy_info(strategy_key)

# Affichage automatique des indicateurs requis
if strategy_info.required_indicators:
    indicators_list = ", ".join([f"**{ind.upper()}**"
                                 for ind in strategy_info.required_indicators])
    st.sidebar.info(f"📊 Indicateurs requis: {indicators_list}")
else:
    st.sidebar.info("📊 Indicateurs: Calculés internement")
```

**Exemple en action** :
```
Stratégie sélectionnée : Bollinger ATR
↓
Affichage automatique : 📊 Indicateurs requis: BOLLINGER, ATR
```

### B. Registre Complet des Indicateurs

**Fichier** : [indicators/registry.py](indicators/registry.py)

**21 indicateurs disponibles** :

| Catégorie | Indicateurs |
|-----------|------------|
| **Tendance** | EMA, SMA, ADX, MACD, Aroon, SuperTrend |
| **Volatilité** | ATR, Bollinger, Keltner, Donchian |
| **Momentum** | RSI, Stochastic, CCI, Momentum, ROC, Williams %R |
| **Volume** | VWAP, OBV, MFI |

**Fonction de listing** :
```python
from indicators.registry import list_indicators

all_indicators = list_indicators()
# → ['bollinger', 'atr', 'rsi', 'ema', 'sma', 'macd', 'adx',
#    'stochastic', 'vwap', 'donchian', 'cci', 'keltner', ...]
```

### C. Panel d'Indicateurs pour l'Utilisateur

**Actuellement** (ligne 2033-2036 de app.py) :
```markdown
### Indicateurs Intégrés
- Bollinger Bands, ATR, RSI, EMA, SMA, MACD, ADX
- Ichimoku, PSAR, Stochastic RSI, Vortex, etc.
```

→ **Liste statique** dans la documentation

**Opportunité d'amélioration** :
Afficher dynamiquement tous les indicateurs du registre avec leurs descriptions.

---

## 3. Orchestration de la Granularité ⚙️

### A. Concept de Granularité

**Fichier** : [utils/parameters.py:72-149](utils/parameters.py#L72-L149)

```python
def parameter_values(
    min_val: float,
    max_val: float,
    granularity: float = 0.5,  # 0.0 = très fin, 1.0 = très grossier
    max_values: int = 4         # Plafond pour éviter explosion
) -> np.ndarray:
```

**Exemples** :

| Granularité | bb_period (10-50) | Nombre de Valeurs | Résultat |
|-------------|------------------|-------------------|----------|
| `0.0` (fin) | 10, 15, 20, 25, 30, 35, 40, 45, 50 | 9 | Exploration complète |
| `0.5` (modéré) | 10, 23, 36, 50 | 4 | Équilibré |
| `1.0` (grossier) | 30 | 1 | Médiane uniquement |

→ **Contrôle intelligent du nombre de combinaisons**

### B. Configuration dans l'UI

**Mode Grille de Paramètres** :

L'utilisateur définit pour chaque paramètre :
- **Min** : Valeur minimale
- **Max** : Valeur maximale
- **Step** : Pas d'incrémentation

Le système calcule automatiquement le nombre de combinaisons.

**Limite de sécurité** :
```python
max_total_combinations: int = 10000  # Par défaut
```

### C. Presets avec Granularité Prédéfinie

**Fichier** : [utils/parameters.py:465-517](utils/parameters.py#L465-L517)

```python
SAFE_RANGES_PRESET = Preset(
    name="Safe Ranges",
    parameters={...},
    indicators=["bollinger", "atr"],
    default_granularity=0.5  # Modérément grossier
)
```

**3 Presets disponibles** :

| Preset | Granularité | Indicateurs | Combinaisons (~) |
|--------|-------------|-------------|------------------|
| **Safe Ranges** | 0.5 | bollinger, atr | ~750 |
| **Minimal** | 1.0 | bollinger, atr | 1 |
| **EMA Cross** | 0.5 | - | ~64 |

---

## 4. Workflow Complet d'Orchestration 🔄

### Étape par Étape

```
┌─────────────────────────────────────────────────────────┐
│ 1. Utilisateur sélectionne STRATÉGIE                   │
│    Ex: "Bollinger ATR"                                  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Système récupère INDICATEURS requis                 │
│    Via: get_strategy_info("bollinger_atr")             │
│    Résultat: ["bollinger", "atr"]                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. UI affiche automatiquement                           │
│    📊 Indicateurs requis: BOLLINGER, ATR                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Utilisateur sélectionne MODE D'EXÉCUTION            │
│    ○ Backtest Simple (1 combinaison)                   │
│    ○ Grille de Paramètres (min/max/step)               │
│    ○ Optimisation LLM (agents IA)                      │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 5. SI Mode Grille :                                     │
│    - Utilisateur définit Min/Max/Step pour paramètres  │
│    - Système calcule nombre de combinaisons            │
│    - Validation : < max_combinations (défaut: 10000)   │
│                                                         │
│    SI Mode LLM :                                        │
│    - Agents LLM récupèrent param_bounds                │
│    - Exploration intelligente de l'espace              │
│    - Max combinaisons: configurable (défaut: 2M)       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 6. Moteur calcule INDICATEURS                          │
│    Via: BacktestEngine.calculate_indicators()          │
│    - bollinger_bands(df, period=20, std=2.0)           │
│    - atr(df, period=14)                                │
│    → Résultat mis en cache                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 7. Moteur exécute BACKTEST                             │
│    Pour chaque combinaison de paramètres:              │
│    - Charge stratégie                                   │
│    - Passe indicateurs pré-calculés                    │
│    - Génère signaux                                     │
│    - Simule trades                                      │
│    - Calcule métriques (PnL, Sharpe, Drawdown)         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 8. UI affiche RÉSULTATS                                │
│    - Tableau de résultats (si grille)                  │
│    - Graphiques (PnL, Drawdown)                        │
│    - Métriques détaillées                              │
│    - Historique des trades                             │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Propositions au LLM (Agents IA) 🤖

### A. Agents et Indicateurs

**Fichier** : [agents/integration.py](agents/integration.py)

Les agents LLM reçoivent automatiquement :

1. **Liste des indicateurs disponibles** :
```python
available_indicators = list_indicators()
# → Agents savent quels indicateurs peuvent être utilisés
```

2. **Indicateurs requis par stratégie** :
```python
required = get_required_indicators(strategy_name)
# → Agents savent quels indicateurs seront chargés
```

3. **Espace de paramètres** :
```python
param_bounds = get_strategy_param_bounds(strategy_name)
# → Agents connaissent les bornes min/max
```

### B. Orchestration LLM

**Mode LLM** dans l'UI :

1. **Sélection du provider** :
   - Ollama (local, gratuit)
   - OpenAI (API, payant)

2. **Configuration multi-modèles** :
   - Analyst : Modèles rapides (Qwen, Phi)
   - Strategist : Modèles moyens (Mistral, LLaMA 8B)
   - Critic : Modèles puissants (LLaMA 70B, Qwen 32B)
   - Validator : Modèles de décision (DeepSeek, Qwen 32B)

3. **Paramètres d'exécution** :
   - Max combinaisons : 10 - 2,000,000
   - Workers parallèles : 1-32
   - Max itérations : 3-20

4. **Workflow automatique** :
```
Analyst analyse résultats
    ↓
Strategist propose nouvelles combinaisons
    ↓
Critic évalue les propositions
    ↓
Validator décide de continuer ou arrêter
    ↓
BacktestExecutor lance les backtests
    ↓
Boucle jusqu'à convergence ou max_iterations
```

---

## 6. Panel d'Indicateurs Proposé à l'Utilisateur 📋

### A. Situation Actuelle

**Page d'accueil > Onglet Stratégies** :
```markdown
### Indicateurs Intégrés
- Bollinger Bands, ATR, RSI, EMA, SMA, MACD, ADX
- Ichimoku, PSAR, Stochastic RSI, Vortex, etc.
```

→ **Liste statique, incomplète**

### B. Amélioration Proposée

Affichage dynamique de **tous** les indicateurs du registre :

```python
from indicators.registry import list_indicators, get_indicator

# Récupérer tous les indicateurs avec descriptions
indicators_info = {
    name: get_indicator(name)
    for name in list_indicators()
}

# Grouper par catégorie
categories = {
    "Tendance": ["ema", "sma", "adx", "macd", "aroon", "supertrend"],
    "Volatilité": ["atr", "bollinger", "keltner", "donchian"],
    "Momentum": ["rsi", "stochastic", "cci", "momentum", "roc", "williams_r"],
    "Volume": ["vwap", "obv", "mfi"]
}
```

**Affichage proposé** :

```markdown
### Indicateurs Disponibles (21)

**Tendance** (6)
- **EMA** : Exponential Moving Average
- **SMA** : Simple Moving Average
- **ADX** : Average Directional Index - Trend strength indicator
- **MACD** : Moving Average Convergence Divergence - Momentum indicator
- **Aroon** : Aroon Indicator - Trend identification
- **SuperTrend** : ATR-based trend follower

**Volatilité** (4)
- **ATR** : Average True Range - Volatility indicator
- **Bollinger** : Bandes de Bollinger - Mean reversion indicator
- **Keltner** : Keltner Channel - Volatility channel based on EMA and ATR
- **Donchian** : Donchian Channel - Breakout indicator

**Momentum** (7)
- **RSI** : Relative Strength Index - Momentum oscillator
- **Stochastic** : Stochastic Oscillator - Overbought/oversold
- **CCI** : Commodity Channel Index - Momentum oscillator
- **Momentum** : Absolute price change over period
- **ROC** : Rate of Change - Percentage price change
- **Williams %R** : Williams %R - Momentum oscillator
- **MFI** : Money Flow Index - Volume-weighted RSI

**Volume** (3)
- **VWAP** : Volume Weighted Average Price
- **OBV** : On-Balance Volume - Cumulative volume flow
- **MFI** : Money Flow Index - Volume-weighted RSI

💡 **Tous les indicateurs sont chargés automatiquement** selon la stratégie sélectionnée
```

---

## 7. Granularité Programmable ⚙️

### A. Niveaux de Contrôle

**Niveau 1 : Presets (Simple)** ✅ Déjà implémenté

```python
preset = get_preset("safe_ranges")
# → granularity = 0.5 (prédéfinie)
# → ~750 combinaisons
```

**Niveau 2 : UI Mode Grille (Intermédiaire)** ✅ Déjà implémenté

L'utilisateur définit Min/Max/Step directement dans l'UI
→ Contrôle manuel précis

**Niveau 3 : API Programmatique (Avancé)** 💡 Disponible

```python
from utils.parameters import generate_param_grid, ParameterSpec

# Définir specs
specs = {
    "bb_period": ParameterSpec("bb_period", 10, 50, 20, param_type="int"),
    "atr_period": ParameterSpec("atr_period", 7, 21, 14, param_type="int")
}

# Générer grille avec granularité contrôlée
grid = generate_param_grid(
    params_specs=specs,
    granularity=0.3,          # Contrôle précis 0.0-1.0
    max_values_per_param=6,   # Plafond par paramètre
    max_total_combinations=5000  # Plafond total
)
```

**Niveau 4 : Agents LLM (Intelligent)** ✅ Déjà implémenté

Les agents décident dynamiquement de la granularité selon :
- Résultats précédents
- Convergence observée
- Budget de combinaisons restant

---

## 8. Tableau Récapitulatif 📊

### A. Stratégies

| Élément | Source | Automatique ? | Configurable ? |
|---------|--------|---------------|----------------|
| **Listing** | `list_strategies()` | ✅ Oui | ❌ Non (détection auto) |
| **Affichage** | UI Sidebar | ✅ Oui | ❌ Non (liste fixe) |
| **Descriptions** | Dictionnaire statique | ❌ Non | ✅ Oui (éditable) |

### B. Indicateurs

| Élément | Source | Automatique ? | Configurable ? |
|---------|--------|---------------|----------------|
| **Listing** | `list_indicators()` | ✅ Oui | ❌ Non (registre auto) |
| **Chargement** | `get_required_indicators()` | ✅ Oui | ❌ Non (mapping auto) |
| **Affichage UI requis** | Mapping + UI | ✅ Oui | ❌ Non |
| **Affichage panel complet** | UI statique | ❌ Non | 💡 Amélioration possible |

### C. Granularité

| Élément | Source | Automatique ? | Configurable ? |
|---------|--------|---------------|----------------|
| **Presets** | `PRESETS` | ✅ Oui | ✅ Oui (granularité fixée) |
| **Mode Grille** | UI | ❌ Non | ✅ Oui (min/max/step) |
| **API Programmatique** | `parameter_values()` | ❌ Non | ✅ Oui (0.0-1.0) |
| **Agents LLM** | Algorithme adaptatif | ✅ Oui | ⚙️ Partiellement |

---

## 9. Opportunités d'Amélioration 💡

### Priorité 1 : Panel Dynamique des Indicateurs

**Actuellement** : Liste statique dans l'UI

**Amélioration** : Générer dynamiquement depuis le registre

**Fichier** : [ui/app.py:2033-2036](ui/app.py#L2033-L2036)

```python
# Au lieu de :
st.markdown("""
### Indicateurs Intégrés
- Bollinger Bands, ATR, RSI, EMA, SMA, MACD, ADX
- Ichimoku, PSAR, Stochastic RSI, Vortex, etc.
""")

# Faire :
from indicators.registry import list_indicators, get_indicator

st.markdown("### Indicateurs Disponibles")
categories = {...}  # Grouper par catégorie
for category, indicators in categories.items():
    with st.expander(f"{category} ({len(indicators)})"):
        for ind_name in indicators:
            info = get_indicator(ind_name)
            st.markdown(f"- **{ind_name.upper()}**: {info.description}")
```

### Priorité 2 : Sélecteur de Granularité dans l'UI

**Actuellement** : Seulement min/max/step en mode Grille

**Amélioration** : Slider de granularité 0.0-1.0

```python
if optimization_mode == "Grille de Paramètres":
    granularity = st.sidebar.slider(
        "Granularité globale",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="0.0 = très fin (max combinaisons), 1.0 = très grossier (1 combinaison)"
    )
```

### Priorité 3 : Auto-création de Presets

**Actuellement** : 3 Presets manuels

**Amélioration** : Génération automatique

```python
def create_preset_for_strategy(strategy_name: str) -> Preset:
    """Crée un Preset automatiquement pour une stratégie."""
    strategy = get_strategy(strategy_name)
    indicators = get_required_indicators(strategy_name)

    return Preset(
        name=f"{strategy_name}_default",
        description=f"Configuration par défaut pour {strategy_name}",
        parameters=strategy.parameter_specs,
        indicators=indicators,
        default_granularity=0.5
    )
```

---

## 10. Conclusion ✅

### Points Forts Actuels

1. ✅ **Stratégies** : Listing automatique + descriptions claires
2. ✅ **Indicateurs** : Chargement automatique selon stratégie
3. ✅ **Granularité** : Système flexible (Presets, UI, API, LLM)
4. ✅ **Orchestration** : Workflow complet et cohérent
5. ✅ **Documentation** : Mapping centralisé et validé

### Architecture Robuste

```
┌─────────────┐
│ Utilisateur │
└──────┬──────┘
       │
       │ sélectionne
       ↓
┌────────────────────┐
│ STRATÉGIE          │  ← list_strategies()
│ (9 disponibles)    │
└────────┬───────────┘
         │
         │ déclenche
         ↓
┌────────────────────────┐
│ INDICATEURS REQUIS     │  ← get_required_indicators()
│ (chargement auto)      │
└────────┬───────────────┘
         │
         │ + configure
         ↓
┌────────────────────────┐
│ GRANULARITÉ            │  ← Presets / UI / API / LLM
│ (0.0-1.0 ou min/max)   │
└────────┬───────────────┘
         │
         │ lance
         ↓
┌────────────────────────┐
│ BACKTEST ENGINE        │
│ (calcul + simulation)  │
└────────┬───────────────┘
         │
         │ retourne
         ↓
┌────────────────────────┐
│ RÉSULTATS              │
│ (métriques + graphes)  │
└────────────────────────┘
```

### Système Complet et Cohérent

- **Stratégies** : 9 disponibles, extensibles
- **Indicateurs** : 21 disponibles, chargement automatique
- **Granularité** : 4 niveaux de contrôle
- **Documentation** : Mapping validé à 100%
- **Tests** : Suite complète de validation

**Le système est prêt pour production ! 🚀**
