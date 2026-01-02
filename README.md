# backtest_core

Point d entree unique pour les agents (LLM ou humains).
Lire AGENTS.md en premier pour les regles et le journal des modifications.

- [AGENTS.md](AGENTS.md)

## Arborescence du projet (auto)
Ce bloc est regenere par `python tools/update_readme_tree.py`.

<!-- TREE:START -->
```
backtest_core/
|-- agents/
|   |-- __init__.py
|   |-- analyst.py
|   |-- autonomous_strategist.py
|   |-- backtest_executor.py
|   |-- base_agent.py
|   |-- critic.py
|   |-- indicator_context.py
|   |-- integration.py
|   |-- llm_client.py
|   |-- model_config.py
|   |-- ollama_manager.py
|   |-- orchestration_logger.py
|   |-- orchestrator.py
|   |-- state_machine.py
|   |-- strategist.py
|   \-- validator.py
|-- backtest/
|   |-- __init__.py
|   |-- engine.py
|   |-- errors.py
|   |-- execution.py
|   |-- execution_fast.py
|   |-- facade.py
|   |-- metrics_tier_s.py
|   |-- monte_carlo.py
|   |-- optuna_optimizer.py
|   |-- pareto.py
|   |-- performance.py
|   |-- simulator.py
|   |-- simulator_fast.py
|   |-- storage.py
|   |-- sweep.py
|   \-- validation.py
|-- cli/
|   |-- __init__.py
|   \-- commands.py
|-- config/
|   \-- indicator_ranges.toml
|-- data/
|-- docs/
|-- indicators/
|   |-- __init__.py
|   |-- adx.py
|   |-- amplitude_hunter.py
|   |-- aroon.py
|   |-- atr.py
|   |-- bollinger.py
|   |-- cci.py
|   |-- donchian.py
|   |-- ema.py
|   |-- fear_greed.py
|   |-- fibonacci.py
|   |-- ichimoku.py
|   |-- keltner.py
|   |-- macd.py
|   |-- mfi.py
|   |-- momentum.py
|   |-- obv.py
|   |-- onchain_smoothing.py
|   |-- pi_cycle.py
|   |-- pivot_points.py
|   |-- psar.py
|   |-- registry.py
|   |-- roc.py
|   |-- rsi.py
|   |-- standard_deviation.py
|   |-- stoch_rsi.py
|   |-- stochastic.py
|   |-- supertrend.py
|   |-- volume_oscillator.py
|   |-- vortex.py
|   |-- vwap.py
|   \-- williams_r.py
|-- performance/
|   |-- __init__.py
|   |-- benchmark.py
|   |-- device_backend.py
|   |-- gpu.py
|   |-- memory.py
|   |-- monitor.py
|   |-- parallel.py
|   \-- profiler.py
|-- strategies/
|   |-- __init__.py
|   |-- base.py
|   |-- bollinger_atr.py
|   |-- bollinger_atr_v2.py
|   |-- bollinger_atr_v3.py
|   |-- ema_cross.py
|   |-- indicators_mapping.py
|   |-- macd_cross.py
|   \-- rsi_reversal.py
|-- templates/
|   |-- analyst.jinja2
|   |-- critic.jinja2
|   |-- strategist.jinja2
|   \-- validator.jinja2
|-- tests/
|-- tools/
|   |-- analyze_cprofile_stats.py
|   |-- check_gpu.py
|   |-- configure_ollama_multigpu.py
|   |-- debug_sharpe_calculation.py
|   |-- diagnose_bollinger.py
|   |-- diagnose_metrics.py
|   |-- diagnose_sharpe_anomaly.py
|   |-- generate_6month_data.py
|   |-- profile.bat
|   |-- profile.ps1
|   |-- profile_analyzer.py
|   |-- profile_backtest_cprofile.py
|   |-- profile_demo.py
|   |-- profile_metrics.py
|   |-- profiler.py
|   |-- reorganize_root.py
|   |-- run_atr_grid_mini.py
|   |-- run_profile_big.py
|   |-- setup_llama33_70b.py
|   |-- test_all_fixes.py
|   |-- test_bollinger_visualization.py
|   |-- test_cpu_gpu_parallel.py
|   |-- test_equity_mtm.py
|   |-- test_grid_search_trades.py
|   |-- test_grid_ui_simulation.py
|   |-- test_httpx_streamlit.py
|   |-- test_llama33_70b.py
|   |-- test_llama33_backtest.py
|   |-- test_multigpu_realtime.py
|   |-- test_sharpe_fix.py
|   |-- test_sharpe_realdata.py
|   |-- test_streamlit_crash.py
|   |-- test_worker_pool.py
|   |-- update_readme_tree.py
|   |-- validate_backtest_integrity.py
|   \-- verify_ui_imports.py
|-- ui/
|   |-- components/
|   |   |-- archive/
|   |   |-- __init__.py
|   |   |-- agent_timeline.py
|   |   |-- charts.py
|   |   |-- model_selector.py
|   |   |-- monitor.py
|   |   |-- sweep_monitor.py
|   |   \-- validation_viewer.py
|   |-- __init__.py
|   |-- app.py
|   |-- constants.py
|   |-- context.py
|   |-- deep_trace_viewer.py
|   |-- emergency_stop.py
|   |-- helpers.py
|   |-- indicators_panel.py
|   |-- log_taps.py
|   |-- main.py
|   |-- model_presets.py
|   |-- orchestration_viewer.py
|   |-- results.py
|   |-- sidebar.py
|   |-- state.py
|   \-- validation_integration.py
|-- utils/
|   |-- __init__.py
|   |-- checkpoint.py
|   |-- circuit_breaker.py
|   |-- config.py
|   |-- data.py
|   |-- error_recovery.py
|   |-- gpu_oom.py
|   |-- gpu_utils.py
|   |-- health.py
|   |-- indicator_ranges.py
|   |-- llm_memory.py
|   |-- log.py
|   |-- memory.py
|   |-- model_loader.py
|   |-- observability.py
|   |-- parameters.py
|   |-- preset_validation.py
|   |-- run_tracker.py
|   |-- session_param_tracker.py
|   |-- session_ranges_tracker.py
|   |-- template.py
|   |-- version.py
|   \-- visualization.py
|-- AGENTS.md
|-- analyze_all_results.py
|-- backtest_core.code-workspace
|-- CLAUDE.md
|-- install.bat
|-- LICENSE
|-- metrics_types.py
|-- Modelfile.llama33-multigpu
|-- pyproject.toml
|-- pytest-watch.ini
|-- README.md
|-- requirements-gpu.txt
|-- requirements-performance.txt
|-- requirements.txt
|-- restart_ollama_multigpu.bat
|-- run_grid_backtest.py
|-- run_llm_optimization.py
|-- run_streamlit.bat
|-- run_streamlit_with_logs.bat
|-- Start-OllamaMultiGPU.ps1
\-- t_core
```
<!-- TREE:END -->
