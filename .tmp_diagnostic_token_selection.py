#!/usr/bin/env python3
"""
Script de diagnostic : Analyser la sélection des tokens dans l'historique Builder
"""
import json
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any

def analyze_builder_history():
    """Analyse l'historique du Builder pour diagnostiquer le biais de sélection."""

    # Chemins possibles pour l'historique
    history_files = [
        Path("sandbox_strategies/_exploration_state.json"),
        Path(".tmp/builder_autonomous_history.json"),
    ]

    history: List[Dict[str, Any]] = []

    # Charger l'historique depuis les fichiers
    for file_path in history_files:
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        history.extend(data)
                    elif isinstance(data, dict) and "history" in data:
                        history.extend(data["history"])
            except Exception as e:
                print(f"⚠️  Erreur lecture {file_path}: {e}")

    if not history:
        print("❌ Aucun historique trouvé.")
        print("   Vérifiez que le Builder autonome a déjà tourné et généré des sessions.")
        return

    print(f"📊 Analyse de {len(history)} sessions Builder\n")
    print("=" * 80)

    # ── Statistiques par symbole ──
    symbols = [h.get("symbol") for h in history if h.get("symbol")]
    timeframes = [h.get("timeframe") for h in history if h.get("timeframe")]
    pairs = [(h.get("symbol"), h.get("timeframe")) for h in history
             if h.get("symbol") and h.get("timeframe")]

    symbol_counts = Counter(symbols)
    timeframe_counts = Counter(timeframes)
    pair_counts = Counter(pairs)

    print("## Distribution des SYMBOLES")
    print("-" * 80)
    total_symbols = sum(symbol_counts.values())
    for sym, count in symbol_counts.most_common(20):
        pct = 100 * count / total_symbols if total_symbols > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{sym:15} {count:4} ({pct:5.1f}%) {bar}")

    print("\n## Distribution des TIMEFRAMES")
    print("-" * 80)
    total_tfs = sum(timeframe_counts.values())
    for tf, count in timeframe_counts.most_common(20):
        pct = 100 * count / total_tfs if total_tfs > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"{tf:15} {count:4} ({pct:5.1f}%) {bar}")

    print("\n## Top 15 COUPLES (symbol, timeframe)")
    print("-" * 80)
    total_pairs = sum(pair_counts.values())
    for (sym, tf), count in pair_counts.most_common(15):
        pct = 100 * count / total_pairs if total_pairs > 0 else 0
        print(f"{sym:12} {tf:8} {count:4} ({pct:5.1f}%)")

    # ── Analyse de la diversité récente (sliding window de 6) ──
    print("\n## Analyse de la DIVERSITÉ récente (fenêtre de 6 runs)")
    print("-" * 80)

    if len(pairs) >= 6:
        recent_windows = []
        for i in range(len(pairs) - 5):
            window = pairs[i:i+6]
            unique_symbols = len(set(s for s, _ in window))
            unique_timeframes = len(set(tf for _, tf in window))
            unique_pairs = len(set(window))
            recent_windows.append({
                "index": i,
                "unique_symbols": unique_symbols,
                "unique_timeframes": unique_timeframes,
                "unique_pairs": unique_pairs,
                "window": window,
            })

        # Moyenne de diversité
        avg_symbols = sum(w["unique_symbols"] for w in recent_windows) / len(recent_windows)
        avg_timeframes = sum(w["unique_timeframes"] for w in recent_windows) / len(recent_windows)
        avg_pairs = sum(w["unique_pairs"] for w in recent_windows) / len(recent_windows)

        print(f"Moyenne de symboles uniques par fenêtre de 6: {avg_symbols:.2f} / 6")
        print(f"Moyenne de timeframes uniques par fenêtre de 6: {avg_timeframes:.2f} / 6")
        print(f"Moyenne de couples uniques par fenêtre de 6: {avg_pairs:.2f} / 6")

        # Pires fenêtres (plus de répétitions)
        worst_windows = sorted(recent_windows, key=lambda w: w["unique_pairs"])[:5]
        print("\n🔴 Pires 5 fenêtres (le moins de diversité):")
        for w in worst_windows:
            print(f"  Runs {w['index']}-{w['index']+5}: "
                  f"{w['unique_pairs']} couples uniques, "
                  f"{w['unique_symbols']} symboles uniques")
            for idx, (s, tf) in enumerate(w["window"]):
                print(f"    [{w['index']+idx}] {s} {tf}")

    # ── Répétitions consécutives ──
    print("\n## RÉPÉTITIONS CONSÉCUTIVES du même couple")
    print("-" * 80)
    consecutive_runs = []
    current_pair = None
    current_count = 0
    current_start = 0

    for idx, pair in enumerate(pairs):
        if pair == current_pair:
            current_count += 1
        else:
            if current_count >= 2:
                consecutive_runs.append({
                    "pair": current_pair,
                    "count": current_count,
                    "start_index": current_start,
                })
            current_pair = pair
            current_count = 1
            current_start = idx

    if current_count >= 2:
        consecutive_runs.append({
            "pair": current_pair,
            "count": current_count,
            "start_index": current_start,
        })

    if consecutive_runs:
        print("🔴 Séquences de répétitions trouvées:")
        for run in sorted(consecutive_runs, key=lambda r: r["count"], reverse=True)[:10]:
            sym, tf = run["pair"]
            print(f"  {sym:12} {tf:8} → {run['count']} fois consécutives "
                  f"(runs {run['start_index']}-{run['start_index']+run['count']-1})")
    else:
        print("✅ Aucune répétition consécutive du même couple détectée.")

    # ── Diagnostics clés ──
    print("\n## DIAGNOSTIC")
    print("=" * 80)

    if len(symbol_counts) == 0:
        print("❌ AUCUN symbole dans l'historique → Problème d'enregistrement")
    elif len(symbol_counts) == 1:
        print(f"❌ UN SEUL symbole utilisé ({list(symbol_counts.keys())[0]}) → "
              "auto_market_pick probablement désactivé OU univers restreint")
    elif len(symbol_counts) <= 3:
        print(f"⚠️  Très peu de symboles différents ({len(symbol_counts)}) → "
              "Vérifier l'univers all_symbols dans la sidebar")

    # Vérifier si un symbole domine >50%
    if symbol_counts:
        most_common_sym, most_common_count = symbol_counts.most_common(1)[0]
        dominance = 100 * most_common_count / total_symbols
        if dominance > 50:
            print(f"🔴 BIAIS MAJEUR: {most_common_sym} représente {dominance:.1f}% des runs")
            print(f"   → Vérifier que recent_markets est bien passé au LLM")
            print(f"   → Vérifier les logs du prompt LLM (shuffled_symbols, diversity_instruction)")
        elif dominance > 33:
            print(f"⚠️  BIAIS MODÉRÉ: {most_common_sym} représente {dominance:.1f}% des runs")
        else:
            print(f"✅ Distribution acceptable: symbole max = {dominance:.1f}%")

    # Vérifier l'évolution récente (10 derniers runs)
    if len(pairs) >= 10:
        recent_10_symbols = [s for s, _ in pairs[-10:]]
        recent_10_unique = len(set(recent_10_symbols))
        print(f"\n📊 Diversité des 10 derniers runs: {recent_10_unique}/10 symboles uniques")
        if recent_10_unique == 1:
            print("   🔴 CRITIQUE: Le même symbole est utilisé sur tous les 10 derniers runs")
        elif recent_10_unique <= 3:
            print("   ⚠️  FAIBLE: Peu de variation récente")
        else:
            print("   ✅ CORRECT: Bonne rotation observée")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC: Sélection des tokens dans Strategy Builder")
    print("=" * 80 + "\n")
    analyze_builder_history()
    print("\n" + "=" * 80)
    print("Analyse terminée.")
    print("=" * 80 + "\n")
