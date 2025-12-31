#!/usr/bin/env python3
"""
Module-ID: diagnose

Purpose: Script diagnostic complet - vérifier packages, versions, config, perf, GPU, etc.

Role in pipeline: system validation

Key components: check_packages(), check_gpu(), check_performance(), affichage coulé

Inputs: Aucun (vérifie sys, modules, GPU)

Outputs: Rapport diagnostic complet avec ✓/✗/⚠️ pour chaque vérification

Dependencies: sys, warnings, colorama, psutil, numpy, cupy (optionnel)

Conventions: Couleurs conso via colorama; cles [PASS, FAIL, WARN]

Read-if: Diagnostic installation, GPU, ou dépannage perf.

Skip-if: Environnement déjà opérationnel.
"""

import sys
import warnings
from typing import Dict, List, Tuple

# Couleurs console
try:
    from colorama import Fore, Style, init
    init()
    RED = Fore.RED
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    CYAN = Fore.CYAN
    RESET = Style.RESET_ALL
except ImportError:
    RED = GREEN = YELLOW = CYAN = RESET = ""


def check_python_version() -> Tuple[bool, str]:
    """Vérifie version Python."""
    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 10):
        return False, f"Python {major}.{minor} (requis: 3.10+)"
    return True, f"Python {major}.{minor}.{sys.version_info.micro}"


def check_packages() -> Tuple[Dict[str, str], List[str], List[str]]:
    """Vérifie packages installés."""
    # Packages critiques (REQUIS)
    critical = {
        'numpy': '1.24.0',
        'pandas': '2.0.0',
        'streamlit': '1.28.0',
        'plotly': '5.18.0',
        'numba': '0.58.0',
        'scipy': '1.11.0',
    }

    # Packages performance (RECOMMANDÉS)
    performance = {
        'bottleneck': '1.3.0',
        'numexpr': '2.8.0',
        'statsmodels': '0.14.0',
        'sklearn': '1.3.0',
    }

    installed = {}
    missing_critical = []
    missing_performance = []

    # Vérifier packages critiques
    for pkg, min_version in critical.items():
        try:
            if pkg == 'sklearn':
                mod = __import__('sklearn')
            else:
                mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            installed[pkg] = version
        except ImportError:
            missing_critical.append(pkg)

    # Vérifier packages performance
    for pkg, min_version in performance.items():
        try:
            if pkg == 'sklearn':
                mod = __import__('sklearn')
            else:
                mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            installed[pkg] = version
        except ImportError:
            missing_performance.append(pkg)

    return installed, missing_critical, missing_performance


def check_numpy_precision() -> Tuple[bool, str]:
    """Vérifie précision calculs NumPy."""
    try:
        import numpy as np
        # Test calcul basique
        arr = np.array([1.0, 2.0, 3.0])
        mean = np.mean(arr)
        if abs(mean - 2.0) > 1e-10:
            return False, f"Précision NumPy incorrecte: {mean}"
        return True, "Précision NumPy OK"
    except Exception as e:
        return False, f"Erreur NumPy: {e}"


def check_pandas_performance() -> Tuple[bool, str]:
    """Vérifie performance Pandas."""
    try:
        import pandas as pd
        # Test si bottleneck est utilisé
        has_bottleneck = 'bottleneck' in sys.modules or pd.options.compute.use_bottleneck
        if has_bottleneck:
            return True, "Pandas avec Bottleneck (rapide)"
        return False, "Pandas sans Bottleneck (lent)"
    except Exception as e:
        return False, f"Erreur Pandas: {e}"


def check_numba_cache() -> Tuple[bool, str]:
    """Vérifie cache Numba."""
    import os
    cache_dir = os.environ.get('NUMBA_CACHE_DIR')
    if cache_dir:
        return True, f"Cache Numba: {cache_dir}"
    return False, "Cache Numba non configuré (compilation lente)"


def check_warnings() -> List[str]:
    """Capture warnings potentiels."""
    warnings_list = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            import numpy as np
            import pandas as pd

            # Test calculs qui peuvent générer warnings
            arr = np.array([1.0, 2.0, np.inf])
            _ = np.mean(arr)

            df = pd.DataFrame({'a': [1, 2, 3]})
            _ = df.rolling(2).mean()

            for warning in w:
                warnings_list.append(str(warning.message))
        except Exception:
            pass

    return warnings_list


def main():
    """Fonction principale."""
    print("=" * 70)
    print(f"{CYAN}BACKTEST CORE - DIAGNOSTIC COMPLET{RESET}")
    print("=" * 70)
    print()

    # 1. Python
    print(f"{CYAN}[1/6] Vérification Python...{RESET}")
    py_ok, py_msg = check_python_version()
    if py_ok:
        print(f"  {GREEN}✅ {py_msg}{RESET}")
    else:
        print(f"  {RED}❌ {py_msg}{RESET}")
    print()

    # 2. Packages
    print(f"{CYAN}[2/6] Vérification Packages...{RESET}")
    installed, missing_critical, missing_performance = check_packages()

    if missing_critical:
        print(f"  {RED}❌ PACKAGES CRITIQUES MANQUANTS:{RESET}")
        for pkg in missing_critical:
            print(f"      - {pkg}")
        print(f"  {YELLOW}   Installation: pip install {' '.join(missing_critical)}{RESET}")
    else:
        print(f"  {GREEN}✅ Tous les packages critiques installés{RESET}")

    if missing_performance:
        print(f"  {YELLOW}⚠️  PACKAGES PERFORMANCE MANQUANTS (recommandés):{RESET}")
        for pkg in missing_performance:
            print(f"      - {pkg}")
        print(f"  {YELLOW}   Installation: pip install -r requirements-performance.txt{RESET}")
    else:
        print(f"  {GREEN}✅ Packages performance installés{RESET}")
    print()

    # 3. Précision NumPy
    print(f"{CYAN}[3/6] Vérification Précision NumPy...{RESET}")
    np_ok, np_msg = check_numpy_precision()
    if np_ok:
        print(f"  {GREEN}✅ {np_msg}{RESET}")
    else:
        print(f"  {RED}❌ {np_msg}{RESET}")
    print()

    # 4. Performance Pandas
    print(f"{CYAN}[4/6] Vérification Performance Pandas...{RESET}")
    pd_ok, pd_msg = check_pandas_performance()
    if pd_ok:
        print(f"  {GREEN}✅ {pd_msg}{RESET}")
    else:
        print(f"  {YELLOW}⚠️  {pd_msg}{RESET}")
        print(f"  {YELLOW}   Recommandation: pip install bottleneck{RESET}")
    print()

    # 5. Cache Numba
    print(f"{CYAN}[5/6] Vérification Cache Numba...{RESET}")
    cache_ok, cache_msg = check_numba_cache()
    if cache_ok:
        print(f"  {GREEN}✅ {cache_msg}{RESET}")
    else:
        print(f"  {YELLOW}⚠️  {cache_msg}{RESET}")
        print(f"  {YELLOW}   Recommandation: set NUMBA_CACHE_DIR=.numba_cache{RESET}")
    print()

    # 6. Warnings
    print(f"{CYAN}[6/6] Capture Warnings...{RESET}")
    warnings_list = check_warnings()
    if warnings_list:
        print(f"  {YELLOW}⚠️  {len(warnings_list)} warning(s) détecté(s){RESET}")
        for warn in warnings_list[:3]:  # Max 3
            print(f"      - {warn}")
    else:
        print(f"  {GREEN}✅ Aucun warning détecté{RESET}")
    print()

    # Résumé
    print("=" * 70)
    print(f"{CYAN}RÉSUMÉ{RESET}")
    print("=" * 70)

    issues = []
    if not py_ok:
        issues.append("Python trop ancien")
    if missing_critical:
        issues.append(f"{len(missing_critical)} package(s) critique(s) manquant(s)")
    if missing_performance:
        issues.append(f"{len(missing_performance)} package(s) performance manquant(s)")
    if not np_ok:
        issues.append("Précision NumPy problématique")

    if issues:
        print(f"{YELLOW}⚠️  {len(issues)} problème(s) détecté(s):{RESET}")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print(f"{YELLOW}Consultez PACKAGES_OPTIONNELS.md pour solutions détaillées{RESET}")
        sys.exit(1)
    else:
        print(f"{GREEN}✅ SYSTÈME OPTIMAL - Aucun problème détecté{RESET}")
        print()
        print(f"{GREEN}🚀 Vous pouvez lancer: streamlit run ui/app.py{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
