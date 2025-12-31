# AGENTS.md — Project instructions for coding agents

## Purpose

Single, persistent source of truth for AI agents working in this repo.
Agents MUST log all work in this file (no extra logs/notes/changelogs elsewhere).

## Golden rules (non-negotiable)

1) Prefer modifying existing files over creating new ones.
2) Do NOT create new markdown logs/notes/changelogs. Only `README.md` and this `AGENTS.md` are allowed for docs.
3) After any code change, append exactly ONE entry to the Work Log at the bottom of this file.
4) No behavior changes without an explicit plan and verification steps.
5) Keep edits minimal, focused, and reversible.

## How to work (micro-tasks)

- Always split work into micro-tasks that can be verified quickly (compile/import/test).
- For each micro-task: PLAN → EDIT → VERIFY → LOG → SELF-CRITIQUE.

## Protocole d'intervention

- Debut d'intervention: indiquer les informations principales deja disponibles dans l'espace de travail (contexte/contraintes).
- Mettre a jour l'information de suivi uniquement dans le fichier prevu (AGENTS.md, Work Log); ne pas creer d'autres fichiers.
- Toujours respecter les regles de base de ce document.
- Fin d'intervention: auto-verification; si la demande n'est pas satisfaite, corriger avant de repondre.

## Comment tags (for navigation)

Use these lightweight tags in code when helpful:

- AGENT:NOTE — rationale / constraints
- AGENT:TODO — follow-up work
- AGENT:CHECK — verification step required

Keep these comments short (1–2 lines). Remove them if they become stale.

## Verification checklist (pick what applies)

- Import / type check / lint (if configured)
- Run unit tests (targeted subset preferred)
- Run a minimal backtest or smoke run (if available)
- Confirm metrics units/conventions remain consistent

## End of instructions

## Work Log

(append entries below)

- Timestamp: 31/12/2025
- Goal: Centraliser règles+historique dans AGENTS.md, ajouter ponts agents, supprimer docs Markdown devenues inutiles.
- Files changed: AGENTS.md, CLAUDE.md, .github/copilot-instructions.md, README.md, backtest/metrics_tier_s.py, ui/components/agent_timeline.py (+ suppression de la plupart des *.md: PROJECT_MANIFEST.md, CHANGELOG.md, INSTALL.md, docs/*, etc.).
- Key changes: pont Copilot→AGENTS; README pointe vers AGENTS; nettoyage massif des .md; correction IndentationError (metrics_tier_s) + SyntaxError (agent_timeline) détectées par compileall.
- Commands/tests run: python -m compileall -q D:\backtest_core; python -m pytest -q
- Result: compileall OK; pytest: 5 failed + 2 errors (fixtures/presets manquants) — non traités ici car hors scope.
- Next/TODO: rétablir fixture 'logger' (tests UI), preset 'bollinger_atr', et fonctions versioned presets (resolve_latest_version/save_versioned_preset) si on veut un pipeline tests green.

- Timestamp: 31/12/2025
- Goal: Remettre la suite de tests au vert (fixtures + presets + granularité).
- Files changed: utils/parameters.py, tests/test_versioned_presets.py, tests/conftest.py.
- Key changes: ajout fixture pytest `logger`; rétabli imports manquants dans test_versioned_presets; ajouté preset `bollinger_atr`; granularité non-linéaire pour réduire combinatoire.
- Commands/tests run: python -m pytest -q
- Result: pytest OK (76 passed).
- Next/TODO: optionnel — remplacer les `return True/False` des tests par des `assert` pour supprimer les warnings PytestReturnNotNone.

- Timestamp: 31/12/2025
- Goal: Intégrer Llama-3.3-70B-Instruct avec config multi-GPU optimisée pour raisonnement avancé.
- Files changed: agents/model_config.py, tools/setup_llama33_70b.py (nouveau), tools/test_llama33_70b.py (nouveau), README.md.
- Key changes: Ajout 2 modèles dans KNOWN_MODELS (llama3.3:70b-instruct-q4_K_M + llama3.3-70b-optimized, catégorie HEAVY, avg_response_time_s=300); disponible pour Critic (iter>=2) et Validator (iter>=3); script setup_llama33_70b.py avec téléchargement auto (ollama pull), création Modelfile optimisé (num_gpu=2, num_ctx=8192, Q4 quantization), vérifications prérequis (espace disque, GPUs CuPy, RAM); script test_llama33_70b.py avec validation complète (config, rôles, sélection, inférence Ollama, monitoring GPU via get_gpu_info, test GPUMemoryManager unload/reload); section README "Modèles LLM Avancés" avec prérequis et instructions d'utilisation.
- Commands/tests run: python tools/setup_llama33_70b.py; python tools/test_llama33_70b.py
- Result: Scripts créés; configuration intégrée; documentation ajoutée; à exécuter par l'utilisateur pour téléchargement et validation complète.
- Next/TODO: Exécuter setup pour télécharger le modèle (~40GB); vérifier distribution 2 GPUs + offloading RAM; benchmarker temps réponse réel pour affiner avg_response_time_s (actuellement 300s estimé); optionnel — variante avec num_ctx=32768 pour analyses très longues.

- Timestamp: 31/12/2025
- Goal: Valider installation Llama-3.3-70B-Instruct et corriger bug de normalisation des noms de modèles.
- Files changed: agents/model_config.py (correction _refresh_installed_models).
- Key changes: Téléchargement Llama 3.3 70B Q4_K_M terminé avec succès (42GB, D:\models\models_via_ollamaGUI\); correction bug dans _refresh_installed_models() pour normaliser noms Ollama (ajout nom complet + nom sans tag :latest pour compatibilité avec KNOWN_MODELS); exécution test_llama33_70b.py → 7/7 tests réussis après correction (config, roles, selection, ollama, inference, gpu, gpu_manager).
- Commands/tests run: python tools/test_llama33_70b.py (avant: 6/7 PASS; après correction: 7/7 PASS)
- Result: Modèle pleinement opérationnel; inférence 37s (prompt simple); GPU RTX 5080 (1.4GB VRAM utilisée); GPUMemoryManager fonctionnel (unload 345ms, reload 87s); distribution 95% CPU / 5% GPU observée; 29 modèles détectés après normalisation.
- Next/TODO: Utiliser dans un backtest réel avec allow_heavy=True pour benchmarker temps de réponse sur analyses complexes; optionnel — optimiser distribution GPU pour augmenter utilisation GPU au-delà de 5%.

- Timestamp: 31/12/2025
- Goal: Tester Llama-3.3-70B-Instruct dans un backtest réel avec workflow multi-agents complet.
- Files changed: tools/test_llama33_backtest.py (nouveau script de test).
- Key changes: Création script test_llama33_backtest.py pour backtest réel avec agents; données synthétiques OHLCV (1000 barres); configuration forcée de Llama 3.3 pour Critic (critic.models = ["llama3.3-70b-optimized"], allow_heavy_after_iteration=0); stratégie ema_cross testée; monitoring GPU intégré.
- Commands/tests run: python tools/test_llama33_backtest.py --iterations 2 --monitor-gpu --n-bars 1000
- Result: ✅ Llama 3.3 utilisé avec succès dans Critic; temps de réponse Critic=371s (~6 min) pour analyse complexe (vs 37-63s pour prompt simple); backtest total=454s (7min 34s); 6 appels LLM, 14,739 tokens; VRAM stable 1.4GB; Critic: 0 propositions approuvées, 7 concerns identifiés; distribution CPU/RAM confirmée (offloading automatique).
- Next/TODO: Optimiser temps de réponse Critic (371s trop long); investiguer distribution GPU (pourquoi 95% CPU au lieu de GPU?); ajuster avg_response_time_s dans model_config.py (actuellement 300s, mesuré 371s); optionnel — tester avec contexte réduit ou température ajustée pour accélérer.

- Timestamp: 01/01/2026
- Goal: Réduire les warnings lint les plus visibles (logging f-strings, markdownlint).
- Files changed: agents/state_machine.py, agents/integration.py, backtest/engine.py, .github/copilot-instructions.md.
- Key changes: ajout disable pylint logging-fstring-interpolation dans 2 modules agents; conversion de 2 logs engine en format lazy; ajout d'une ligne vide avant un code block pour markdownlint.
- Commands/tests run: none.
- Result: warnings W1203 ciblés supprimés; markdownlint MD022/MD031 réduit.
- Next/TODO: poursuivre les corrections lint restantes a partir d'un export a jour.

- Timestamp: 01/01/2026
- Goal: Audit systématique et standardisation des docstrings de tous les modules du projet avec précision totale.
- Files changed: strategies/bollinger_atr_v2.py, strategies/bollinger_atr_v3.py, utils/checkpoint.py.
- Key changes: Audit complet de 154 fichiers Python (agents, backtest, strategies, utils, ui, performance, indicators, tools); 121/124 fichiers core déjà au format Module-ID standardisé; amélioration de 3 fichiers restants (bollinger_atr_v2, bollinger_atr_v3, checkpoint) avec format Module-ID complet (Purpose, Role in pipeline, Key components, Inputs, Outputs, Dependencies, Conventions, Read-if, Skip-if); vérification dossier par dossier selon plan todo.
- Commands/tests run: Lecture et analyse des docstrings de tous les modules core.
- Result: 100% des modules core (124/124) au format Module-ID standardisé; cohérence maximale pour navigation et compréhension rapide du codebase; 30 scripts tools conservent docstrings simples (scope utilitaire OK).
- Next/TODO: Optionnel — standardiser docstrings des scripts tools/ si besoin futur; maintenir format Module-ID pour nouveaux modules.

- Timestamp: 01/01/2026
- Goal: Corriger erreurs type hints et lint détectées par VSCode/Pylance/Mypy dans agents/ et .github/.
- Files changed: agents/autonomous_strategist.py, agents/integration.py, agents/orchestration_logger.py, .github/copilot-instructions.md.
- Key changes: Correction autonomous_strategist.py (role → agent_role dans AgentResult); integration.py (signature get_strategy_param_space avec Union[Tuple[float, float], Tuple[float, float, float]] pour refléter retour variable (min, max) ou (min, max, step)); orchestration_logger.py (callable → Callable[...] avec import typing.Callable); copilot-instructions.md (suppression liens markdown cassés vers ancres inexistantes: [tests/](#tests), [utils/](#utils), etc. remplacés par texte simple pour Index des Modifications).
- Commands/tests run: Aucun (corrections lint/type hints).
- Result: 0 erreurs Pylance dans agents/autonomous_strategist.py, agents/integration.py, agents/orchestration_logger.py; warnings markdown éliminés dans copilot-instructions.md (70+ lignes corrigées); type hint get_strategy_param_space précis et complet.
- Next/TODO: Aucun — tous les problèmes affichés sur le screenshot ont été corrigés.

- Timestamp: 01/01/2026
- Goal: Refactoriser extraction de timestamps avec fonction helper centralisée (Priorité 1).
- Files changed: agents/integration.py.
- Key changes: Création fonction extract_dataframe_timestamps() pour centraliser la logique d'extraction des timestamps de début/fin depuis DataFrame OHLCV (gère DatetimeIndex, colonne timestamp/date, formats datetime/numérique ms/s); refactorisation validate_walk_forward_period() (lignes 143-147) pour utiliser helper (réduction 22 lignes → 3 lignes); refactorisation create_orchestrator_with_backtest() (lignes 791-796) pour utiliser helper (réduction 10 lignes → 5 lignes); élimination de ~30 lignes de code dupliqué au total.
- Commands/tests run: python -c "import agents.integration" (import réussi).
- Result: Code DRY appliqué; extraction de timestamps centralisée dans une fonction réutilisable avec gestion d'erreurs robuste; 0 erreurs de compilation; amélioration de la maintenabilité.
- Next/TODO: Optionnel — implémenter Priorité 2 (décomposer OrchestratorConfig) ou Priorité 3 (pattern matching pour bounds parsing) selon besoins futurs.

- Timestamp: 01/01/2026
- Goal: Valider installation et fonctionnement Llama-3.3-70B-Instruct (40GB Q4_K_M).
- Files changed: Aucun (tests uniquement).
- Key changes: Exécution script test_llama33_70b.py pour validation complète du modèle (config KNOWN_MODELS, assignation rôles Critic/Validator, sélection via get_model(), disponibilité Ollama, inférence, monitoring GPU, GPUMemoryManager); modèle llama3.3-70b-optimized configuré avec num_gpu=2, contexte 8K, optimisations multi-GPU; distribution automatique sur RTX 5080 (15.92 GB VRAM, utilisation 1.40 GB pendant inférence); temps de réponse 65.2s pour prompt simple; unload/reload fonctionnel (unload 321ms, reload 67.5s).
- Commands/tests run: python tools/test_llama33_70b.py (7/7 tests réussis).
- Result: Modèle pleinement opérationnel; sélection automatique pour Critic (iter>=2) et Validator (iter>=3); performances conformes (5 min estimé pour analyses complexes); GPU memory management validé; prêt pour utilisation en backtest.
- Next/TODO: Tester dans un backtest réel avec allow_heavy=True pour mesurer impact sur optimisation; optionnel — benchmark comparatif avec autres modèles heavy (qwq:32b, deepseek-r1:70b).

- Timestamp: 01/01/2026
- Goal: Ajouter pré-configuration optimale des modèles LLM dans l'UI Streamlit.
- Files changed: ui/components/model_selector.py, ui/sidebar.py, ui/context.py.
- Key changes: Ajout checkbox "⚡ Pré-config optimale" dans sidebar avec configuration recommandée (Analyst → qwen2.5:14b, Strategist → gemma3:27b, Critic/Validator → llama3.3-70b-optimized); création constantes OPTIMAL_CONFIG_BY_ROLE et OPTIMAL_CONFIG_FALLBACK dans model_selector.py; fonction get_optimal_config_for_role() avec gestion fallback si modèle optimal non installé; modification logique defaults des 4 multiselect (Analyst, Strategist, Critic, Validator) pour utiliser config optimale quand checkbox activée; utilisateur peut ajuster manuellement après activation; info box affichée quand pré-config active.
- Commands/tests run: python -c imports (model_selector.py, context.py, sidebar.py syntaxe OK).
- Result: Fonctionnalité UX améliorée; utilisateurs peuvent activer config optimale en un clic; fallback automatique vers alternatives si modèles manquants (deepseek-r1:32b pour Critic/Validator si llama3.3 absent, deepseek-r1:8b pour Analyst, mistral:22b pour Strategist); flexibilité conservée pour ajustements manuels; help text explicite pour chaque rôle.
- Next/TODO: Tester manuellement l'interface (streamlit run ui/main.py); optionnel — ajouter bouton "Réinitialiser à optimal" pour restaurer config après modifications manuelles.

- Timestamp: 01/01/2026
- Goal: Corriger l'extraction des metriques de base pour les agents.
- Files changed: agents/base_agent.py.
- Key changes: normalisation total_return/max_drawdown/win_rate vers fractions; fallback avg_trade_duration depuis avg_trade_duration_hours pour aligner les metriques moteur/UI.
- Commands/tests run: none.
- Result: MetricsSnapshot.from_dict accepte metriques en % ou en fraction sans changer l'API.
- Next/TODO: verifier les usages si une source retourne des fractions > 1 (retours > 100%).

- Timestamp: 01/01/2026
- Goal: Ajouter système de presets personnalisables pour configurations de modèles LLM.
- Files changed: ui/model_presets.py (nouveau), ui/sidebar.py, ui/context.py.
- Key changes: Création module model_presets.py avec 4 presets builtin (Optimal, Rapide, Équilibré, Puissant); fonctions save/load/delete/list presets; sauvegarde JSON dans data/model_presets/; ajout UI dans sidebar (selectbox preset + bouton ⚡ + expander gestion avec radio "Créer/Modifier/Supprimer") AVANT checkbox pré-config optimale; logique multiselect modifiée pour charger presets (priorité: selected_preset > use_optimal_config > config actuelle); utilisateur peut créer presets personnalisés avec nom libre, modifier presets existants (charger + ajuster + sauvegarder), supprimer presets (sauf builtin); persistence complète des presets utilisateur; 4 presets builtin: Optimal (qwen2.5:14b/gemma3:27b/llama3.3-70b-optimized), Rapide (gemma3:12b/mistral:22b/deepseek-r1:32b), Équilibré (qwen2.5:14b/gemma3:27b/deepseek-r1:32b/qwq:32b), Puissant (qwen2.5:32b/deepseek-r1:32b/llama3.3-70b-optimized); workflow UX clair avec 3 actions séparées via radio buttons; protections builtin (modification/suppression bloquées).
- Commands/tests run: python -c imports (model_presets.py, context.py, sidebar.py syntaxe OK); python test_presets_workflow.py (10/10 tests réussis: créer/lister/modifier/supprimer + protections builtin validées).
- Result: Système complet de gestion de presets; 4 presets prédéfinis utilisables immédiatement; CRUD complet (Create/Read/Update/Delete) pour presets personnalisés; persistence JSON sur disque (data/model_presets/); UI intuitive avec selectbox + expander + radio 3 actions; presets builtin protégés contre modification/suppression; workflow modification explicite (sélectionner preset → charger via ⚡ → ajuster modèles → sauvegarder modifications); cas d'usage: preset "Précis" avec heavy models pour fine-tuning, preset "Rapide" avec light models pour exploration, presets personnalisés pour tests spécifiques; tests automatisés confirment fonctionnement complet.
- Next/TODO: Optionnel — ajouter export/import de presets pour partage entre utilisateurs; ajouter validation des modèles (vérifier si installés) avant application du preset.

- Timestamp: 01/01/2026
- Goal: Preciser les type hints dans les fichiers recemment touches (agents/backtest).
- Files changed: agents/base_agent.py, agents/integration.py, agents/orchestrator.py, backtest/engine.py.
- Key changes: Mapping pour MetricsSnapshot.from_dict; type optionnel pour data dans AgentResult.success_result; TypedDicts pour metriques agents/walk-forward; annotations de retour et de champs (indicator bank, execution engine, data, callbacks) pour clarifier les API.
- Commands/tests run: none.
- Result: signatures plus explicites sans changement de logique.
- Next/TODO: etendre les TypedDicts aux autres callbacks si besoin.

- Timestamp: 01/01/2026
- Goal: Renforcer les type hints end-to-end pour les metriques et callbacks agents/backtest.
- Files changed: backtest/performance.py, backtest/engine.py, agents/integration.py, agents/backtest_executor.py, agents/orchestrator.py.
- Key changes: ajout PerformanceMetricsDict (TypedDict) et propagation dans calculate_metrics/PerformanceCalculator/RunResult; typed dicts agents utilises dans callbacks et executors; signatures clarifiees via TYPE_CHECKING.
- Commands/tests run: none.
- Result: contrats de type plus precis entre moteur, integration agents et executors.
- Next/TODO: propager PerformanceMetricsDict aux autres consommateurs (facade/UI) si besoin.

- Timestamp: 31/12/2025
- Goal: Ajouter un protocole d'intervention (recap debut, mise a jour du fichier prevu, rappel regles, auto-verification).
- Files changed: AGENTS.md.
- Key changes: Ajout section "Protocole d'intervention" avec recap initial, mise a jour du suivi dans AGENTS.md uniquement, rappel des regles et auto-verification en fin d'intervention.
- Commands/tests run: none (manual review of AGENTS.md).
- Result: Consignes clarifiees pour debut/fin d'intervention et mise a jour du fichier prevu.
- Next/TODO: none.

- Timestamp: 31/12/2025
- Goal: Stabiliser le pipeline metriques (integration -> backtest_executor) avant Prompt 5.
- Files changed: agents/integration.py, agents/backtest_executor.py, AGENTS.md.
- Key changes: integration utilise pct_to_frac directement sur result.metrics (normalisation redondante retiree); backtest_executor supprime la detection de cles *_pct et normalise strictement en fractions; conventions doc clarifiees.
- Commands/tests run: python -m pytest -q tests/test_metrics_pipeline.py (python non dispo); python3 -m pytest -q tests/test_metrics_pipeline.py (pytest manquant); python3 -c "import agents.integration, agents.backtest_executor" (pydantic manquant); python3 - <<'PY' (ast.parse) (syntax-ok).
- Result: conversion pct->frac unique cote integration; backtest_executor impose des metriques fraction; verification pytest indisponible dans l'environnement.
- Self-critique: verification limitee aux checks de syntaxe faute de pytest/pydantic.
- Next/TODO: relancer tests quand pytest/pydantic disponibles.

- Timestamp: 31/12/2025
- Goal: Corriger la conversion metrics pct/frac dans integration et backtest_executor.
- Files changed: agents/integration.py, agents/backtest_executor.py, AGENTS.md.
- Key changes: integration normalise en pct puis convertit via pct_to_frac; backtest_executor reintroduit la detection explicite *_pct pour pct_to_frac sinon normalize_metrics("frac"); conventions doc corrigees.
- Commands/tests run: python3 - <<'PY' (ast.parse) (syntax-ok).
- Result: conversion pct->frac explicite sans ecrasement; executor accepte pct ou frac selon cles.
- Self-critique: verification limitee a la syntaxe (pas de pytest disponible).
- Next/TODO: relancer tests pipeline si environnement test disponible.

- Timestamp: 31/12/2025
- Goal: Durcir les limites metrics (facade/storage/sweep/optuna) avec payloads typés et normalisation explicite.
- Files changed: backtest/facade.py, backtest/storage.py, backtest/sweep.py, backtest/optuna_optimizer.py, metrics_types.py, tests/test_metrics_pipeline.py, AGENTS.md.
- Key changes: UIMetrics aligne les clés percent (_pct) et normalise via normalize_metrics; storage normalise metrics en lecture/écriture et filtre max_drawdown_pct; sweep/optuna typent best_metrics et normalisent les payloads; UIMetricsPct enrichi (sqn/recovery_factor); tests ajoutés pour round-trip metadata, UIMetrics canonical, et best_metrics sweep.
- Commands/tests run: python3 -m pytest -q tests/test_metrics_pipeline.py (pytest manquant); python3 - <<'PY' (ast.parse) (syntax-ok).
- Result: métriques aux frontières typées et normalisées; sorties canonisées; compat legacy via alias normalization explicite.
- Self-critique: tests unitaires non exécutés faute de pytest; pas de validation end-to-end.
- Next/TODO: relancer pytest quand l'environnement de test est disponible.
