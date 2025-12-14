# Système de Templates Jinja2 pour Prompts LLM

> **Documentation technique** : Centralisation des prompts LLM avec moteur de templates Jinja2

---

## 📋 Sommaire

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
3. [Utilisation](#utilisation)
4. [Templates disponibles](#templates-disponibles)
5. [Filtres personnalisés](#filtres-personnalisés)
6. [Exemples](#exemples)
7. [Tests](#tests)
8. [Avantages](#avantages)

---

## Vue d'ensemble

Le système de templates centralise la génération de tous les prompts LLM, séparant le texte du code pour améliorer la maintenabilité et la clarté.

**Avant** : 
```python
def _build_analysis_prompt(self, context):
    prompt = f"Analyze the following...\n"
    prompt += f"Strategy: {context.strategy_name}\n"
    # 50+ lignes de concaténation...
```

**Après** :
```python
def _build_analysis_prompt(self, context):
    return render_prompt("analyst.jinja2", {
        "strategy_name": context.strategy_name,
        # ... contexte structuré
    })
```

---

## Architecture

```
backtest_core/
├── templates/                    # 🆕 Gabarits Jinja2
│   ├── analyst.jinja2           # Prompt Agent Analyst
│   ├── strategist.jinja2        # Prompt Agent Strategist
│   ├── critic.jinja2            # Prompt Agent Critic
│   └── validator.jinja2         # Prompt Agent Validator
├── utils/
│   └── template.py              # 🆕 Moteur de rendu
└── agents/
    ├── analyst.py               # ✏️ Utilise template
    ├── strategist.py            # ✏️ Utilise template
    ├── critic.py                # ✏️ Utilise template
    └── validator.py             # ✏️ Utilise template
```

---

## Utilisation

### 1. Import de base

```python
from utils.template import render_prompt
```

### 2. Rendu d'un template

```python
# Préparer le contexte
context = {
    "strategy_name": "ema_cross",
    "current_metrics": metrics_snapshot,
    "iteration": 5,
    # ... autres variables
}

# Rendre le template
prompt = render_prompt("analyst.jinja2", context)
```

### 3. Intégration dans les agents

```python
class AnalystAgent(BaseAgent):
    def _build_analysis_prompt(self, context: AgentContext) -> str:
        """Construit le prompt via template Jinja2."""
        
        template_context = {
            "strategy_name": context.strategy_name,
            "current_metrics": context.current_metrics,
            "iteration": context.iteration,
            # ... mapping AgentContext → dict
        }
        
        return render_prompt("analyst.jinja2", template_context)
```

---

## Templates disponibles

### analyst.jinja2

**Usage** : Analyse quantitative de performance

**Variables requises** :
- `strategy_name` : Nom de la stratégie
- `current_metrics` : MetricsSnapshot
- `iteration` : Numéro d'itération
- `current_params` : Dict des paramètres
- `optimization_target` : Métrique cible
- `min_sharpe`, `max_drawdown_limit`, `min_trades`, `max_overfitting_ratio`

**Variables optionnelles** :
- `train_metrics`, `test_metrics` : Walk-forward
- `iteration_history` : Historique des runs

**Exemple** :
```jinja
Strategy: {{ strategy_name }}
Iteration: {{ iteration }}

Current Parameters:
{% for param_name, param_value in current_params.items() %}
  {{ param_name }}: {{ param_value }}
{% endfor %}

{% if current_metrics -%}
{{ current_metrics.to_summary_str() }}
{% endif %}
```

---

### strategist.jinja2

**Usage** : Génération de propositions d'optimisation

**Variables requises** :
- `strategy_name`, `iteration`
- `param_specs` : Liste[ParameterConfig]
- `current_params` : Dict
- `current_metrics` : MetricsSnapshot
- `overfitting_ratio` : Float

**Variables optionnelles** :
- `analyst_report` : Résumé de l'analyse
- `best_metrics`, `best_params` : Meilleure config

**Exemple** :
```jinja
Current Parameters and Constraints:
{% for spec in param_specs %}
  {{ spec.name }}: {{ current_params.get(spec.name) }} (min={{ spec.min_value }}, max={{ spec.max_value }})
{% endfor %}

{% if overfitting_ratio > 1.5 %}
  ⚠️ HIGH overfitting risk!
{% endif %}
```

---

### critic.jinja2

**Usage** : Évaluation critique des propositions

**Variables requises** :
- `strategy_name`, `iteration`
- `strategist_proposals` : Liste[Dict]
- `current_params` : Dict
- `param_specs` : Liste[ParameterConfig]

**Variables optionnelles** :
- `current_metrics` : Baseline
- `analyst_report` : Contexte

**Exemple** :
```jinja
=== PROPOSALS TO EVALUATE ===
{% for proposal in strategist_proposals %}
--- Proposal {{ proposal.get('id') }}: {{ proposal.get('name') }} ---
Parameters:
{% for param_name, param_value in proposal.get('parameters', {}).items() %}
  {{ param_name }}: {{ current_params.get(param_name) }} → {{ param_value }}
{% endfor %}
{% endfor %}
```

---

### validator.jinja2

**Usage** : Décision finale APPROVE/REJECT/ITERATE

**Variables requises** :
- `strategy_name`, `iteration`
- `objective_check` : Dict[str, bool]
- `current_metrics` : MetricsSnapshot

**Variables optionnelles** :
- `strategist_proposals` : Avec évaluations critic
- `critic_concerns` : Liste de warnings
- `iteration_history` : Historique complet

**Exemple** :
```jinja
=== OBJECTIVE CRITERIA CHECK ===
{% for criterion, passed in objective_check.items() %}
  {{ criterion }}: {% if passed %}✅ PASS{% else %}❌ FAIL{% endif %}
{% endfor %}
```

---

## Filtres personnalisés

### format_percent

Formate un float en pourcentage.

```jinja
{{ 0.1523|format_percent }}  {# Output: 15.23% #}
```

### format_float(n)

Formate un float avec n décimales.

```jinja
{{ 1.23456|format_float(3) }}  {# Output: 1.235 #}
```

### format_metrics

Formate un MetricsSnapshot complet (filtre avancé).

```jinja
{{ metrics|format_metrics }}
{# Output:
Performance Metrics:
  Sharpe Ratio: 1.500
  Total Return: 25.00%
  ...
#}
```

---

## Exemples

### Exemple 1 : Template simple

**Template** (`example.jinja2`) :
```jinja
Hello {{ name }}!
Your score: {{ score|format_float(2) }}
```

**Code** :
```python
result = render_prompt("example.jinja2", {
    "name": "Agent",
    "score": 1.23456
})
# Output: "Hello Agent!\nYour score: 1.23"
```

---

### Exemple 2 : Boucles et conditions

**Template** :
```jinja
Parameters:
{% for param_name, value in params.items() %}
  {{ param_name }}: {{ value }}
{% endfor %}

{% if overfitting_ratio > 1.5 %}
⚠️ High overfitting risk!
{% else %}
✅ Overfitting under control
{% endif %}
```

**Code** :
```python
result = render_prompt("example.jinja2", {
    "params": {"fast": 10, "slow": 20},
    "overfitting_ratio": 1.8
})
```

---

### Exemple 3 : Template from string (pour tests)

```python
from utils.template import render_prompt_from_string

template = "{{ strategy }} iteration {{ iter }}"
result = render_prompt_from_string(template, {
    "strategy": "ema_cross",
    "iter": 5
})
# Output: "ema_cross iteration 5"
```

---

## Tests

### Test du moteur de base

```python
def test_jinja_env_initialization():
    env = get_jinja_env()
    assert env is not None
    assert "format_percent" in env.filters
```

### Test d'un template

```python
def test_analyst_template_renders():
    context = {
        "strategy_name": "ema_cross",
        "iteration": 1,
        # ... autres variables
    }
    result = render_prompt("analyst.jinja2", context)
    assert "ema_cross" in result
    assert "Iteration: 1" in result
```

### Test d'intégration

```python
def test_analyst_uses_template():
    agent = AnalystAgent(llm_config)
    prompt = agent._build_analysis_prompt(context)
    assert len(prompt) > 0
```

**Résultats** :
- ✅ 7 tests moteur de base
- ✅ 6 tests template analyst
- ✅ 5 tests template strategist
- ✅ 4 tests template critic
- ✅ 7 tests template validator
- ✅ 1 test intégration agent

**Total : 30 tests, 100% pass**

---

## Avantages

### 1. Séparation des responsabilités

- **Code** : Logique de traitement
- **Templates** : Texte des prompts
- Modification des prompts sans toucher au code Python

### 2. Maintenabilité

**Avant** :
```python
# 50 lignes de concaténation dispersées dans chaque agent
prompt = "Analyze...\n"
prompt += f"Strategy: {name}\n"
# ...
```

**Après** :
```python
# 1 ligne dans agent + template centralisé
return render_prompt("analyst.jinja2", context)
```

### 3. Réutilisabilité

- Templates partagés entre agents si besoin
- Filtres Jinja2 réutilisables
- Logique de formatting centralisée

### 4. Testabilité

- Tests unitaires sur templates isolés
- Mocking facile du contexte
- Validation de la structure du prompt

### 5. Lisibilité

**Template Jinja2** :
```jinja
{% if train_metrics and test_metrics -%}
Walk-Forward Analysis:
  Train Sharpe: {{ train_metrics.sharpe_ratio|format_float(3) }}
  Test Sharpe: {{ test_metrics.sharpe_ratio|format_float(3) }}
{% endif %}
```

**vs concaténation Python** :
```python
if context.train_metrics and context.test_metrics:
    prompt_parts.append("Walk-Forward Analysis:")
    prompt_parts.append(f"  Train Sharpe: {context.train_metrics.sharpe_ratio:.3f}")
    # ...
```

### 6. Évolutivité

- Ajout de nouveaux templates trivial
- Modification de format sans risque de régression
- Version control friendly (diff lisibles)

---

## Migration depuis ancien système

### Étapes effectuées

1. ✅ Création du module `utils/template.py`
2. ✅ Création de 4 templates (`analyst`, `strategist`, `critic`, `validator`)
3. ✅ Modification des 4 agents pour utiliser `render_prompt()`
4. ✅ Création de 30 tests complets
5. ✅ Ajout de Jinja2 dans `requirements.txt`

### Metrics

| Métrique | Avant | Après | Amélioration |
|----------|-------|-------|--------------|
| **Lignes de code prompt** | ~200 | ~50 | **-75%** |
| **Fichiers prompt** | 4 (dans agents) | 4 (templates/) | Centralisé |
| **Tests prompts** | 0 | 30 | **+∞** |
| **Maintenabilité** | Faible | Élevée | ✅ |

### Rétrocompatibilité

✅ **100% compatible** : Tous les tests existants passent sans modification.

Les méthodes `_build_*_prompt()` conservent leur signature, seule l'implémentation interne change.

---

## Dépendances

```bash
pip install jinja2>=3.1.0
```

Ou via `requirements.txt` :
```txt
jinja2>=3.1.0  # Moteur de templates pour prompts LLM
```

---

*Documentation générée le 13/12/2025*  
*Version : 1.4.0 (Système de templates Jinja2)*
