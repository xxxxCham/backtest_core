#!/usr/bin/env python3
"""
Module-ID: verify_install

Purpose: Vérifier installation complète de toutes dépendances (core, UI, perf, testing, LLM).

Role in pipeline: installation validation

Key components: check_module(), affichage résultat, colorisation

Inputs: Aucun (vérifie sys.modules)

Outputs: Rapport d'installation (core, UI, perf, LLM) avec ✓/✗

Dependencies: sys, importlib, colorama (optionnel)

Conventions: Dépendances par catégorie (Core, UI, Perf, LLM, Testing)

Read-if: Installation première ou dépannage imports manquants.

Skip-if: Environnement déjà vérifié.
"""

import sys
from typing import List, Tuple

# Liste complète des dépendances à vérifier
DEPENDENCIES = [
    # Core
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("pytz", "PyTZ"),
    # UI & Visualization
    ("streamlit", "Streamlit"),
    ("plotly", "Plotly"),
    ("matplotlib", "Matplotlib"),
    ("jinja2", "Jinja2"),
    # Performance
    ("joblib", "Joblib"),
    ("psutil", "PSUtil"),
    ("rich", "Rich"),
    ("numba", "Numba"),
    # Testing
    ("pytest", "PyTest"),
    # Data
    ("pyarrow", "PyArrow"),
    # Optional
    ("pydantic", "Pydantic"),
    ("scipy", "SciPy"),
    ("httpx", "HTTPX"),
    ("optuna", "Optuna"),
]


def check_imports() -> Tuple[List[str], List[str]]:
    """Vérifie l'import de toutes les dépendances."""
    success = []
    failed = []

    for module, name in DEPENDENCIES:
        try:
            __import__(module)
            success.append((module, name))
        except ImportError:
            failed.append((module, name))

    return success, failed


def check_python_version() -> bool:
    """Vérifie la version de Python."""
    major, minor = sys.version_info[:2]
    required_major, required_minor = 3, 10

    if major < required_major or (major == required_major and minor < required_minor):
        print(f"❌ Python {major}.{minor} détecté")
        print(f"⚠️  Python {required_major}.{required_minor}+ requis")
        return False

    print(f"✅ Python {major}.{minor}.{sys.version_info.micro}")
    return True


def get_versions(modules: List[Tuple[str, str]]) -> None:
    """Affiche les versions des modules installés."""
    print("\n📦 Versions des packages:")
    print("-" * 50)

    for module_name, display_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "version inconnue")
            print(f"  {display_name:20s} {version}")
        except Exception:
            pass


def main():
    """Fonction principale."""
    print("=" * 60)
    print("  BACKTEST CORE - VÉRIFICATION DE L'INSTALLATION")
    print("=" * 60)
    print()

    # Vérifier Python
    print("🐍 Vérification de Python...")
    python_ok = check_python_version()
    print()

    # Vérifier les imports
    print("📦 Vérification des dépendances...")
    success, failed = check_imports()

    if success:
        print(f"\n✅ {len(success)}/{len(DEPENDENCIES)} packages installés avec succès:")
        for module, name in success:
            print(f"  ✓ {name}")

    if failed:
        print(f"\n❌ {len(failed)} package(s) manquant(s):")
        for module, name in failed:
            print(f"  ✗ {name} (module: {module})")
        print("\n⚠️  Pour les installer:")
        print("    pip install -r requirements.txt")

    # Afficher les versions
    if success:
        get_versions(success)

    # Résumé
    print("\n" + "=" * 60)
    if not python_ok:
        print("❌ ÉCHEC: Python trop ancien")
        sys.exit(1)
    elif failed:
        print(f"⚠️  ATTENTION: {len(failed)} package(s) manquant(s)")
        sys.exit(1)
    else:
        print("✅ SUCCÈS: Toutes les dépendances sont installées!")
        print()
        print("🚀 Vous pouvez lancer l'interface:")
        print("    streamlit run ui/app.py")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
