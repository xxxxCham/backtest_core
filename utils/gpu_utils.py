"""
Module-ID: utils.gpu_utils

Purpose: Utilitaires GPU - conversions CuPy↔NumPy, détection GPU, interop CPU/GPU.

Role in pipeline: performance

Key components: ensure_numpy_array, ensure_cupy_array, is_gpu_available, ArrayBackend

Inputs: Array NumPy ou CuPy, flag force_cpu

Outputs: Converted array (NumPy ou CuPy), device info

Dependencies: numpy, cupy (optionnel), importlib

Conventions: Détection CuPy runtime; conversions transparentes; fallback NumPy si pas GPU.

Read-if: Modification conversions, détection GPU.

Skip-if: Vous utilisez juste ensure_numpy_array().
"""

from typing import Any, List, Tuple, Union, Optional
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def ensure_numpy_array(
    arr: Union[np.ndarray, Any, None]
) -> Optional[np.ndarray]:
    """
    Convertit un array CuPy en NumPy, laisse les NumPy arrays inchangés.

    Cette fonction détecte automatiquement si l'objet est un CuPy array
    (via l'attribut __cuda_array_interface__) et le convertit en NumPy.
    Les NumPy arrays sont retournés tels quels (pas de copie).

    **Cas d'usage:**
    - Garantir compatibilité avec bibliothèques CPU (pandas, matplotlib)
    - Préparer résultats GPU pour sérialisation (pickle, JSON)
    - Interface entre code GPU et stratégies CPU

    **Performance:**
    - Transfert GPU→CPU: ~1-5ms selon taille (overhead négligeable)
    - NumPy→NumPy: Aucune copie (0ms)

    Args:
        arr: Array à convertir (CuPy, NumPy, ou autre)
             - Si CuPy array: converti en NumPy via cp.asnumpy()
             - Si NumPy array: retourné tel quel (pas de copie)
             - Si None: retourne None
             - Si scalaire (int, float): retourné tel quel

    Returns:
        NumPy array (ou None si input est None)

    Raises:
        TypeError: Si l'objet n'est pas convertible

    Examples:
        >>> # CuPy → NumPy
        >>> import cupy as cp
        >>> gpu_arr = cp.array([1.0, 2.0, 3.0])
        >>> cpu_arr = ensure_numpy_array(gpu_arr)
        >>> type(cpu_arr)
        <class 'numpy.ndarray'>

        >>> # NumPy → NumPy (pas de copie)
        >>> np_arr = np.array([1.0, 2.0, 3.0])
        >>> result = ensure_numpy_array(np_arr)
        >>> result is np_arr  # Même objet
        True

        >>> # Tuple de CuPy arrays
        >>> gpu_tuple = (cp.array([1]), cp.array([2]))
        >>> cpu_tuple = ensure_numpy_array(gpu_tuple)
        >>> all(isinstance(a, np.ndarray) for a in cpu_tuple)
        True

        >>> # None → None
        >>> ensure_numpy_array(None) is None
        True

        >>> # Scalaires passent inchangés
        >>> ensure_numpy_array(42)
        42
    """
    # Cas 1: None
    if arr is None:
        return None

    # Cas 2: Scalaires (int, float, bool)
    if isinstance(arr, (int, float, bool, str)):
        return arr

    # Cas 3: Tuple ou liste d'arrays (récursif)
    if isinstance(arr, (tuple, list)):
        converted = [ensure_numpy_array(item) for item in arr]
        return type(arr)(converted)  # Préserve type (tuple vs list)

    # Cas 4: Déjà un NumPy array → retour direct (pas de copie)
    if isinstance(arr, np.ndarray):
        return arr

    # Cas 5: CuPy array (détection via CUDA Array Interface)
    # Ref: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
    if hasattr(arr, '__cuda_array_interface__'):
        if not HAS_CUPY:
            raise RuntimeError(
                "Détection d'un CuPy array mais CuPy n'est pas installé. "
                "Installez CuPy: pip install cupy-cuda12x"
            )
        # Conversion GPU → CPU
        return cp.asnumpy(arr)

    # Cas 6: Objet avec .get() (interface CuPy alternative)
    if hasattr(arr, 'get') and callable(arr.get):
        try:
            return arr.get()  # CuPy arrays ont .get() → NumPy
        except Exception:
            pass  # Fallback vers erreur

    # Cas 7: Objet inconnu → tentative de conversion NumPy
    try:
        return np.asarray(arr)
    except Exception as e:
        raise TypeError(
            f"Impossible de convertir {type(arr)} en NumPy array. "
            f"Types supportés: CuPy array, NumPy array, tuple/list, None. "
            f"Erreur: {e}"
        )


def is_cupy_array(arr: Any) -> bool:
    """
    Vérifie si un objet est un CuPy array.

    Args:
        arr: Objet à vérifier

    Returns:
        True si CuPy array, False sinon

    Examples:
        >>> import cupy as cp
        >>> is_cupy_array(cp.array([1, 2, 3]))
        True

        >>> import numpy as np
        >>> is_cupy_array(np.array([1, 2, 3]))
        False
    """
    return hasattr(arr, '__cuda_array_interface__')


def is_numpy_array(arr: Any) -> bool:
    """
    Vérifie si un objet est un NumPy array.

    Args:
        arr: Objet à vérifier

    Returns:
        True si NumPy array, False sinon

    Examples:
        >>> import numpy as np
        >>> is_numpy_array(np.array([1, 2, 3]))
        True

        >>> is_numpy_array([1, 2, 3])
        False
    """
    return isinstance(arr, np.ndarray)


def get_array_backend(arr: Any) -> str:
    """
    Retourne le backend de l'array ('cupy', 'numpy', 'unknown').

    Args:
        arr: Array à inspecter

    Returns:
        'cupy', 'numpy', ou 'unknown'

    Examples:
        >>> import numpy as np
        >>> get_array_backend(np.array([1, 2, 3]))
        'numpy'

        >>> import cupy as cp
        >>> get_array_backend(cp.array([1, 2, 3]))
        'cupy'

        >>> get_array_backend([1, 2, 3])
        'unknown'
    """
    if is_cupy_array(arr):
        return 'cupy'
    elif is_numpy_array(arr):
        return 'numpy'
    else:
        return 'unknown'


# ======================== Tests Unitaires Intégrés ========================

def _test_ensure_numpy_array():
    """Tests unitaires pour ensure_numpy_array()."""
    print("=" * 70)
    print("TESTS: ensure_numpy_array()")
    print("=" * 70)

    # Test 1: NumPy → NumPy (pas de copie)
    print("\n[Test 1] NumPy array → NumPy (pas de copie)")
    np_arr = np.array([1.0, 2.0, 3.0])
    result = ensure_numpy_array(np_arr)
    assert result is np_arr, "❌ NumPy array devrait être retourné tel quel"
    assert isinstance(result, np.ndarray), "❌ Type incorrect"
    print(f"✅ PASS - Type: {type(result)}, Same object: {result is np_arr}")

    # Test 2: None → None
    print("\n[Test 2] None → None")
    result = ensure_numpy_array(None)
    assert result is None, "❌ None devrait retourner None"
    print(f"✅ PASS - Result: {result}")

    # Test 3: Scalaires
    print("\n[Test 3] Scalaires (int, float)")
    assert ensure_numpy_array(42) == 42, "❌ Scalaire int échoue"
    assert ensure_numpy_array(3.14) == 3.14, "❌ Scalaire float échoue"
    assert ensure_numpy_array(True) is True, "❌ Scalaire bool échoue"
    print(f"✅ PASS - Scalaires: 42, 3.14, True")

    # Test 4: Tuple de NumPy arrays
    print("\n[Test 4] Tuple de NumPy arrays")
    tuple_arr = (np.array([1]), np.array([2]), np.array([3]))
    result = ensure_numpy_array(tuple_arr)
    assert isinstance(result, tuple), "❌ Devrait retourner tuple"
    assert all(isinstance(a, np.ndarray) for a in result), "❌ Éléments pas NumPy"
    print(f"✅ PASS - Tuple de {len(result)} arrays")

    # Test 5: CuPy → NumPy (si disponible)
    if HAS_CUPY:
        print("\n[Test 5] CuPy array → NumPy")
        gpu_arr = cp.array([1.0, 2.0, 3.0])
        cpu_arr = ensure_numpy_array(gpu_arr)
        assert isinstance(cpu_arr, np.ndarray), "❌ Conversion CuPy échouée"
        assert not is_cupy_array(cpu_arr), "❌ Résultat encore sur GPU"
        assert np.allclose(cpu_arr, [1.0, 2.0, 3.0]), "❌ Valeurs incorrectes"
        print(f"✅ PASS - CuPy → NumPy: {cpu_arr}")

        print("\n[Test 6] Tuple de CuPy arrays")
        gpu_tuple = (cp.array([1]), cp.array([2]))
        cpu_tuple = ensure_numpy_array(gpu_tuple)
        assert isinstance(cpu_tuple, tuple), "❌ Devrait retourner tuple"
        assert all(isinstance(a, np.ndarray) for a in cpu_tuple), "❌ Éléments pas NumPy"
        print(f"✅ PASS - Tuple de {len(cpu_tuple)} CuPy arrays converti")
    else:
        print("\n[Test 5-6] CuPy non disponible, tests skippés")

    print("\n" + "=" * 70)
    print("✅ TOUS LES TESTS PASSÉS")
    print("=" * 70)


if __name__ == '__main__':
    # Exécuter les tests si lancé directement
    _test_ensure_numpy_array()
