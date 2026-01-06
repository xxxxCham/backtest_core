"""
Analyseur Visuel de Profiling - Backtest Core

Génère des rapports HTML avec graphiques interactifs pour analyser
les résultats de profiling.

Usage:
    python tools/profile_analyzer.py --report profiling_results/report.prof --output analysis.html
"""

import argparse
import pstats
from pathlib import Path
from typing import Dict, List, Tuple


def extract_bottlenecks(
    report_path: str,
    top_n: int = 50
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Extrait les goulots d'étranglement d'un rapport de profiling.

    Returns:
        (by_cumulative, by_time, by_calls)
    """
    stats = pstats.Stats(report_path)
    stats.strip_dirs()

    # Par temps cumulé
    stats.sort_stats("cumulative")
    cumulative_stats = []
    for func, data in list(stats.stats.items())[:top_n]:
        cumulative_stats.append({
            "function": f"{func[0]}:{func[1]}:{func[2]}",
            "calls": data[0],
            "tottime": data[2],
            "cumtime": data[3],
            "percall_tot": data[2] / data[0] if data[0] > 0 else 0,
            "percall_cum": data[3] / data[0] if data[0] > 0 else 0,
        })

    # Par temps propre
    stats.sort_stats("time")
    time_stats = []
    for func, data in list(stats.stats.items())[:top_n]:
        time_stats.append({
            "function": f"{func[0]}:{func[1]}:{func[2]}",
            "calls": data[0],
            "tottime": data[2],
            "cumtime": data[3],
            "percall_tot": data[2] / data[0] if data[0] > 0 else 0,
            "percall_cum": data[3] / data[0] if data[0] > 0 else 0,
        })

    # Par nombre d'appels
    stats.sort_stats("calls")
    calls_stats = []
    for func, data in list(stats.stats.items())[:30]:
        calls_stats.append({
            "function": f"{func[0]}:{func[1]}:{func[2]}",
            "calls": data[0],
            "tottime": data[2],
            "cumtime": data[3],
            "percall_tot": data[2] / data[0] if data[0] > 0 else 0,
            "percall_cum": data[3] / data[0] if data[0] > 0 else 0,
        })

    return cumulative_stats, time_stats, calls_stats


def generate_html_report(
    report_path: str,
    output_path: str,
    top_n: int = 50
):
    """Génère un rapport HTML interactif."""
    cumulative, by_time, by_calls = extract_bottlenecks(report_path, top_n)

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse Profiling - Backtest Core</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0e1117;
            color: #fafafa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #26a69a;
            border-bottom: 2px solid #26a69a;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #42a5f5;
            margin-top: 40px;
        }}
        .summary {{
            background: #1e272e;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .summary-item {{
            display: inline-block;
            margin-right: 30px;
        }}
        .summary-label {{
            color: #a8b2d1;
            font-size: 0.9em;
        }}
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #26a69a;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: #1e272e;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #26a69a;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #2c3e50;
        }}
        tr:hover {{
            background: #262d35;
        }}
        .function-name {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            color: #ffd700;
        }}
        .metric {{
            text-align: right;
            font-family: 'Consolas', 'Monaco', monospace;
        }}
        .metric-high {{
            color: #ef5350;
            font-weight: bold;
        }}
        .metric-medium {{
            color: #ffa726;
        }}
        .metric-low {{
            color: #66bb6a;
        }}
        .section {{
            margin-bottom: 60px;
        }}
        .warning {{
            background: #ffeb3b;
            color: #000;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
            font-weight: bold;
        }}
        .tip {{
            background: #1e3a5f;
            border-left: 4px solid #42a5f5;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Analyse de Performance - Backtest Core</h1>

        <div class="summary">
            <div class="summary-item">
                <div class="summary-label">Rapport</div>
                <div class="summary-value">{Path(report_path).name}</div>
            </div>
            <div class="summary-item">
                <div class="summary-label">Fonctions analysées</div>
                <div class="summary-value">{len(cumulative)}</div>
            </div>
        </div>

        <div class="warning">
            ⚠️ ZONES D'OPTIMISATION : Les fonctions en rouge sont les principaux goulots d'étranglement
        </div>

        <div class="section">
            <h2>🔥 Top {min(30, len(cumulative))} - Temps Cumulé (avec appels internes)</h2>
            <div class="tip">
                📌 <strong>Temps cumulé</strong> = temps total passé dans la fonction + tout ce qu'elle appelle.
                Ces fonctions sont les <strong>points d'entrée</strong> des zones lentes.
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Fonction</th>
                        <th>Appels</th>
                        <th>Temps Cumulé (s)</th>
                        <th>Par Appel (ms)</th>
                        <th>% Impact</th>
                    </tr>
                </thead>
                <tbody>
"""

    total_cumtime = sum(s["cumtime"] for s in cumulative[:30])
    for i, stat in enumerate(cumulative[:30], 1):
        pct = (stat["cumtime"] / total_cumtime * 100) if total_cumtime > 0 else 0
        metric_class = "metric-high" if pct > 10 else "metric-medium" if pct > 5 else "metric-low"

        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td class="function-name">{stat['function']}</td>
                        <td class="metric">{stat['calls']:,}</td>
                        <td class="metric {metric_class}">{stat['cumtime']:.3f}</td>
                        <td class="metric">{stat['percall_cum']*1000:.2f}</td>
                        <td class="metric {metric_class}">{pct:.1f}%</td>
                    </tr>
"""

    html += f"""
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>⚡ Top {min(30, len(by_time))} - Temps Propre (sans appels internes)</h2>
            <div class="tip">
                📌 <strong>Temps propre</strong> = temps passé UNIQUEMENT dans cette fonction (hors appels).
                Ces fonctions sont les <strong>véritables consommateurs</strong> de temps CPU.
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Fonction</th>
                        <th>Appels</th>
                        <th>Temps Propre (s)</th>
                        <th>Par Appel (ms)</th>
                        <th>% Impact</th>
                    </tr>
                </thead>
                <tbody>
"""

    total_tottime = sum(s["tottime"] for s in by_time[:30])
    for i, stat in enumerate(by_time[:30], 1):
        pct = (stat["tottime"] / total_tottime * 100) if total_tottime > 0 else 0
        metric_class = "metric-high" if pct > 10 else "metric-medium" if pct > 5 else "metric-low"

        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td class="function-name">{stat['function']}</td>
                        <td class="metric">{stat['calls']:,}</td>
                        <td class="metric {metric_class}">{stat['tottime']:.3f}</td>
                        <td class="metric">{stat['percall_tot']*1000:.2f}</td>
                        <td class="metric {metric_class}">{pct:.1f}%</td>
                    </tr>
"""

    html += f"""
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>🔄 Top {min(20, len(by_calls))} - Nombre d'Appels</h2>
            <div class="tip">
                📌 Fonctions appelées très fréquemment. Même une petite optimisation ici peut avoir un grand impact.
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Fonction</th>
                        <th>Appels</th>
                        <th>Temps Total (s)</th>
                        <th>Par Appel (μs)</th>
                    </tr>
                </thead>
                <tbody>
"""

    for i, stat in enumerate(by_calls[:20], 1):
        metric_class = "metric-high" if stat['calls'] > 100000 else "metric-medium" if stat['calls'] > 10000 else "metric-low"

        html += f"""
                    <tr>
                        <td>{i}</td>
                        <td class="function-name">{stat['function']}</td>
                        <td class="metric {metric_class}">{stat['calls']:,}</td>
                        <td class="metric">{stat['tottime']:.3f}</td>
                        <td class="metric">{stat['percall_tot']*1_000_000:.1f}</td>
                    </tr>
"""

    html += """
                </tbody>
            </table>
        </div>

        <div class="tip">
            <h3>💡 Guide d'Optimisation</h3>
            <ol>
                <li><strong>Temps Cumulé élevé</strong> → Identifier les fonctions appelées à l'intérieur (regarder la table "Temps Propre")</li>
                <li><strong>Temps Propre élevé</strong> → Optimiser directement cette fonction (vectorisation, cache, algorithme plus efficace)</li>
                <li><strong>Nombre d'appels élevé</strong> → Peut-on réduire les appels ? (cache, batch processing, lazy evaluation)</li>
            </ol>
            <h3>🎯 Priorités d'Optimisation</h3>
            <ul>
                <li>🔴 <strong>ROUGE</strong> (>10%) : Goulots critiques - optimisation prioritaire</li>
                <li>🟠 <strong>ORANGE</strong> (5-10%) : Impact significatif - optimisation recommandée</li>
                <li>🟢 <strong>VERT</strong> (<5%) : Impact faible - optimisation optionnelle</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

    # Écrire le rapport
    output_file = Path(output_path)
    output_file.write_text(html, encoding="utf-8")
    print(f"✅ Rapport HTML généré: {output_file}")
    print("   📂 Ouvrir dans un navigateur pour voir l'analyse")


def main():
    parser = argparse.ArgumentParser(description="Analyseur visuel de profiling")
    parser.add_argument("--report", required=True, help="Fichier .prof à analyser")
    parser.add_argument("--output", default="profiling_analysis.html", help="Fichier HTML de sortie")
    parser.add_argument("--top-n", type=int, default=50, help="Nombre de fonctions à analyser")

    args = parser.parse_args()

    generate_html_report(args.report, args.output, args.top_n)


if __name__ == "__main__":
    main()
