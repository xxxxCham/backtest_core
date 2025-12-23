# REPORT_TESTS

Environment
- OS: Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39
- Python: 3.12.3 (venv: /mnt/d/backtest_core/.venv)
- Python exec: /mnt/d/backtest_core/.venv/bin/python
- Project root: /mnt/d/backtest_core

Commands Executed (chronological)
- python --version -> FAIL (python not found)
- sudo apt-get update -> FAIL (sudo password required)
- python3 -m venv --without-pip .venv -> OK
- curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py -> OK
- .venv/bin/python /tmp/get-pip.py -> OK
- source .venv/bin/activate; python --version -> OK
- source .venv/bin/activate; python -c "import sys; print(sys.executable)" -> OK
- source .venv/bin/activate; pip --version -> OK
- source .venv/bin/activate; pip list -> OK
- source .venv/bin/activate; python -c "import platform; print(platform.platform())" -> OK
- export PROJECT_ROOT=/mnt/d/backtest_core; echo $PROJECT_ROOT -> OK
- source .venv/bin/activate; pip install -e '.[dev]' -> FAIL (timeout) then OK
- source .venv/bin/activate; python -c "import ui; import backtest; import strategies; print('imports ok')" -> OK (psutil/rich warnings)
- source .venv/bin/activate; python -m compileall -q . -> FAIL (timeout)
- source .venv/bin/activate; python -m compileall -q -x '(/data/|/logs/|/performance/|/\.venv/|orchestration_logs_)' . -> OK
- source .venv/bin/activate; pip install ruff -> OK
- source .venv/bin/activate; ruff check . -> FAIL (3441 lint issues)
- source .venv/bin/activate; python -m pip check -> OK
- source .venv/bin/activate; pytest -q -> FAIL (FileNotFoundError in capture)
- source .venv/bin/activate; PYTEST_ADDOPTS=--capture=no pytest -q -> OK
- source .venv/bin/activate; PYTEST_ADDOPTS=--capture=no pytest -q --maxfail=1 -> OK
- source .venv/bin/activate; PYTEST_ADDOPTS=--capture=no pytest -q -k 'not slow' -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core --help -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core backtest --help -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core optimize --help -> OK (alias sweep)
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core indicators --help -> OK (alias list indicators)
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core sweep --help -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core optuna --help -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core export --help -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core list strategies --json -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core backtest -s ema_cross -d data/sample_data/BTCUSDT_1h_sample.csv -o /tmp/backtest_short.json --format json -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core backtest -s rsi_reversal -d data/sample_data/BTCUSDT_1h_6months.csv -o /tmp/backtest_medium.json --format json -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core backtest -s macd_cross -d data/sample_data/ETHUSDT_1m_sample.csv -o /tmp/backtest_multi.json --format json -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python -m backtest_core export -i /tmp/backtest_short.json -f html -o /tmp/backtest_short.html -> OK
- source .venv/bin/activate; PYTHONPATH=/mnt/d python /tmp/indicator_check.py -> OK

Step Status
- Step A: OK (python via venv; OS info gathered)
- Step B: OK (editable install after venv bootstrap)
- Step C: PARTIAL (compileall OK with exclusions; ruff check FAIL)
- Step D: OK (pytest ok with capture disabled)
- Step E: OK (CLI help works; optimize/indicators aliases added; export help corrected)
- Step F: OK (3 backtests + JSON output + HTML export)
- Step G: OK (indicator alignment check passed)
- Step H: OK (this report)

Corrections Applied
- cli/commands.py: fix JSON strategy listing; correct metric display (percent handling); include period/meta in output; add date filter; add slippage/symbol/timeframe handling; fix sweep/optuna metric keys; add indicators alias command.
- cli/__init__.py: add optimize alias for sweep; add indicators command; add start/end/symbol/timeframe/slippage options; remove pdf from export choices.
- data/loader.py: treat "unnamed: 0" as time column for CSV index.
- backtest/optuna_optimizer.py: use total_return_pct in summary; accept config/symbol/timeframe; pass to engine; honor CLI seed.

Recommendations
- Investigate pytest capture FileNotFoundError (workaround: PYTEST_ADDOPTS=--capture=no).
- Address ruff lint debt (3441 issues; mostly whitespace/import order/unused imports).
- Consider packaging to avoid needing PYTHONPATH=/mnt/d for `python -m backtest_core` from repo root.
- Optional: add psutil and rich to dev extras to remove runtime warnings.
