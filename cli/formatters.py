"""
Module-ID: cli.formatters

Purpose: Formatage et affichage pour CLI - couleurs, tableaux, progress bars, messages.

Role in pipeline: Utilitaires d'affichage pour toutes les commandes CLI.

Key components: Colors, print_header, print_success, print_error, format_table, format_bytes

Dependencies: colorama (optionnel), tqdm (optionnel)

Conventions: Couleurs désactivables via Colors.disable() ou --no-color

Read-if: Modification du style d'affichage CLI.

Skip-if: Utilisation des commandes sans modifier l'affichage.
"""

from typing import List, Optional

# =============================================================================
# GESTION COULEURS (colorama si disponible)
# =============================================================================

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

    class Fore:
        GREEN = RED = YELLOW = CYAN = BLUE = MAGENTA = WHITE = BLACK = RESET = ""

    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


# =============================================================================
# GESTION PROGRESS BARS (tqdm si disponible)
# =============================================================================

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kwargs: x


# =============================================================================
# CLASSE COLORS
# =============================================================================

class Colors:
    """
    Codes couleurs pour terminal avec support colorama.
    Compatibilité Windows améliorée.
    """
    # Utiliser colorama si disponible
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else "\033[0m"
    BOLD = Style.BRIGHT if COLORAMA_AVAILABLE else "\033[1m"
    DIM = Style.DIM if COLORAMA_AVAILABLE else "\033[2m"

    # Couleurs de texte
    RED = Fore.RED if COLORAMA_AVAILABLE else "\033[91m"
    GREEN = Fore.GREEN if COLORAMA_AVAILABLE else "\033[92m"
    YELLOW = Fore.YELLOW if COLORAMA_AVAILABLE else "\033[93m"
    BLUE = Fore.BLUE if COLORAMA_AVAILABLE else "\033[94m"
    MAGENTA = Fore.MAGENTA if COLORAMA_AVAILABLE else "\033[95m"
    CYAN = Fore.CYAN if COLORAMA_AVAILABLE else "\033[96m"
    WHITE = Fore.WHITE if COLORAMA_AVAILABLE else "\033[97m"

    # Combinaisons utiles
    SUCCESS = f"{GREEN}{BOLD}" if COLORAMA_AVAILABLE else "\033[1;92m"
    ERROR = f"{RED}{BOLD}" if COLORAMA_AVAILABLE else "\033[1;91m"
    WARNING = f"{YELLOW}{BOLD}" if COLORAMA_AVAILABLE else "\033[1;93m"
    INFO = f"{CYAN}" if COLORAMA_AVAILABLE else "\033[96m"

    _disabled = False

    @classmethod
    def disable(cls):
        """Désactive les couleurs."""
        cls._disabled = True
        cls.RESET = cls.BOLD = cls.DIM = ""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = ""
        cls.MAGENTA = cls.CYAN = cls.WHITE = ""
        cls.SUCCESS = cls.ERROR = cls.WARNING = cls.INFO = ""

    @classmethod
    def is_disabled(cls) -> bool:
        """Vérifie si les couleurs sont désactivées."""
        return cls._disabled


# =============================================================================
# FONCTIONS D'AFFICHAGE MESSAGES
# =============================================================================

def print_header(text: str, char: str = "="):
    """Affiche un en-tête formaté avec soulignement."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(Colors.CYAN + char * len(text) + Colors.RESET)


def print_success(text: str):
    """Affiche un message de succès avec ✓."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")


def print_error(text: str):
    """Affiche un message d'erreur avec ✗."""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")


def print_warning(text: str):
    """Affiche un avertissement avec ⚠."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")


def print_info(text: str):
    """Affiche une information avec ℹ."""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")


def print_metric(label: str, value: float, color: Optional[str] = None, suffix: str = ""):
    """Affiche une métrique formatée."""
    color = color or Colors.RESET
    print(f"    {label}:  {color}{value}{Colors.RESET}{suffix}")


# =============================================================================
# FORMATAGE TABLEAUX
# =============================================================================

def format_table(headers: List[str], rows: List[List[str]], indent: int = 2) -> str:
    """
    Formate une table en texte aligné.

    Args:
        headers: Liste des en-têtes de colonnes
        rows: Liste des lignes (chaque ligne est une liste de cellules)
        indent: Nombre d'espaces d'indentation

    Returns:
        Table formatée en string
    """
    if not rows:
        return " " * indent + "(aucune donnée)"

    # Calculer largeurs maximales par colonne
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))

    prefix = " " * indent
    lines = []

    # Header
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    lines.append(f"{prefix}{Colors.BOLD}{header_line}{Colors.RESET}")
    lines.append(prefix + "  ".join("-" * w for w in widths))

    # Rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(f"{prefix}{row_line}")

    return "\n".join(lines)


def format_dict_table(data: dict, title: Optional[str] = None, indent: int = 2) -> str:
    """
    Formate un dictionnaire en table clé-valeur.

    Args:
        data: Dictionnaire à formater
        title: Titre optionnel
        indent: Indentation

    Returns:
        Table formatée
    """
    lines = []
    prefix = " " * indent

    if title:
        lines.append(f"{prefix}{Colors.BOLD}{title}{Colors.RESET}")

    max_key_len = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        lines.append(f"{prefix}  {str(key).ljust(max_key_len)}: {value}")

    return "\n".join(lines)


# =============================================================================
# FORMATAGE VALEURS
# =============================================================================

def format_bytes(bytes_count: float) -> str:
    """Formate un nombre de bytes en unité lisible (KB, MB, GB, etc.)."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"


def format_duration(seconds: float) -> str:
    """Formate une durée en format lisible."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}min"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(value: float, decimals: int = 2) -> str:
    """Formate un nombre avec séparateurs de milliers."""
    if abs(value) >= 1000:
        return f"{value:,.{decimals}f}"
    return f"{value:.{decimals}f}"


def format_pnl(pnl: float, period_days: Optional[int] = None) -> str:
    """
    Formate un PnL avec couleur et optionnellement PnL/jour.

    Args:
        pnl: Profit/Loss en valeur absolue
        period_days: Nombre de jours pour calculer PnL/jour

    Returns:
        String formaté avec couleur
    """
    color = Colors.GREEN if pnl >= 0 else Colors.RED
    sign = "+" if pnl >= 0 else ""

    result = f"{color}{sign}${pnl:,.2f}{Colors.RESET}"

    if period_days and period_days > 0:
        daily_pnl = pnl / period_days
        result += f" ({sign}${daily_pnl:,.2f}/jour)"

    return result


def format_percent(value: float, include_sign: bool = True) -> str:
    """Formate un pourcentage avec couleur."""
    color = Colors.GREEN if value >= 0 else Colors.RED
    sign = "+" if value >= 0 and include_sign else ""
    return f"{color}{sign}{value:.2f}%{Colors.RESET}"


# =============================================================================
# PROGRESS BARS
# =============================================================================

def create_progress_bar(iterable, desc: str = "", total: int = None,
                       disable: bool = False, unit: str = "it"):
    """
    Crée une progress bar élégante avec tqdm.

    Args:
        iterable: Itérable à parcourir
        desc: Description de la tâche
        total: Nombre total d'éléments (auto-détecté si None)
        disable: Désactiver la barre
        unit: Unité à afficher

    Returns:
        Itérable wrappé avec progress bar
    """
    if not TQDM_AVAILABLE or disable:
        return iterable

    return tqdm(
        iterable,
        desc=f"{Colors.CYAN}🔄 {desc}{Colors.RESET}",
        total=total,
        unit=unit,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )


# =============================================================================
# FORMATAGE RÉSULTATS BACKTEST
# =============================================================================

def format_backtest_summary(metrics: dict, period_days: Optional[int] = None) -> str:
    """
    Formate un résumé de backtest pour affichage CLI.

    Args:
        metrics: Dictionnaire des métriques
        period_days: Durée en jours pour calcul PnL/jour

    Returns:
        Résumé formaté multi-lignes
    """
    lines = []

    total_pnl = metrics.get('total_pnl', 0)
    sharpe = metrics.get('sharpe_ratio', 0)
    max_dd = metrics.get('max_drawdown_pct', 0)
    win_rate = metrics.get('win_rate_pct', 0)
    trades = metrics.get('total_trades', metrics.get('trades', 0))
    profit_factor = metrics.get('profit_factor', 0)

    # Couleurs selon performance
    pnl_color = Colors.GREEN if total_pnl > 0 else Colors.RED
    sharpe_color = Colors.GREEN if sharpe > 1 else Colors.YELLOW if sharpe > 0 else Colors.RED

    lines.append(f"  {Colors.BOLD}💰 Performance:{Colors.RESET}")

    # PnL avec daily
    pnl_str = f"${total_pnl:,.2f}"
    if period_days and period_days > 0:
        daily = total_pnl / period_days
        pnl_str += f" (${daily:,.2f}/jour)"
    lines.append(f"    P&L Total:     {pnl_color}{pnl_str}{Colors.RESET}")

    lines.append(f"    Sharpe Ratio:  {sharpe_color}{sharpe:.3f}{Colors.RESET}")
    lines.append(f"    Max Drawdown:  {Colors.RED}{abs(max_dd):.2f}%{Colors.RESET}")
    lines.append(f"    Win Rate:      {win_rate:.1f}%")
    lines.append(f"    Profit Factor: {profit_factor:.2f}")
    lines.append(f"    Trades:        {trades}")

    return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "Colors",
    # Constantes
    "COLORAMA_AVAILABLE",
    "TQDM_AVAILABLE",
    "tqdm",
    # Messages
    "print_header",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "print_metric",
    # Tables
    "format_table",
    "format_dict_table",
    # Valeurs
    "format_bytes",
    "format_duration",
    "format_number",
    "format_pnl",
    "format_percent",
    # Progress
    "create_progress_bar",
    # Résultats
    "format_backtest_summary",
]