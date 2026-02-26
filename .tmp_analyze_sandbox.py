#!/usr/bin/env python3
"""Analyser les dossiers sandbox_strategies pour extraire patterns."""
import os
import re
import json
from pathlib import Path
from collections import Counter
from datetime import datetime

def parse_sandbox_folders():
    """Extraire les métadonnées des dossiers sandbox_strategies."""
    sandbox_path = Path("sandbox_strategies")
    if not sandbox_path.exists():
        print("❌ Dossier sandbox_strategies introuvable")
        return

    folders = [f for f in sandbox_path.iterdir() if f.is_dir()]
    print(f"📂 Analyse de {len(folders)} dossiers dans sandbox_strategies/\n")

    results = []
    for folder in folders:
        # Pattern: YYYYMMDD_HHMMSS_<description>
        match = re.match(r"^(\d{8})_(\d{6})_(.+)$", folder.name)
        if not match:
            continue

        date_str, time_str, description = match.groups()
        try:
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            timestamp = None

        # Essayer de lire session_summary.json si présent
        summary_path = folder / "session_summary.json"
        objective = ""
        status = ""
        best_sharpe = None
        symbol_from_file = ""
        timeframe_from_file = ""

        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    objective = data.get("objective", "")
                    status = data.get("status", "")
                    best_sharpe = data.get("best_sharpe")
                    symbol_from_file = data.get("symbol", "")
                    timeframe_from_file = data.get("timeframe", "")
            except Exception:
                pass

        # Tentative d'extraction depuis la description
        desc_upper = description.upper()

        # Pattern pour symboles (ex: 0GUSDC, BTCUSDC, ETHUSDC)
        symbol_match = re.search(r'\b([A-Z0-9]{2,10}USDC)\b', desc_upper)
        symbol_from_desc = symbol_match.group(1) if symbol_match else ""

        # Pattern pour timeframes (ex: 1h, 30m, 1d, 1w, 4h)
        tf_match = re.search(r'\b(\d+[MHDW])\b', desc_upper)
        timeframe_from_desc = tf_match.group(1).lower() if tf_match else ""

        # Priorité: fichier JSON > description
        symbol = symbol_from_file or symbol_from_desc
        timeframe = timeframe_from_file or timeframe_from_desc

        results.append({
            "timestamp": timestamp,
            "folder": folder.name,
            "description": description,
            "objective": objective,
            "status": status,
            "best_sharpe": best_sharpe,
            "symbol": symbol,
            "timeframe": timeframe,
        })

    # Tri chronologique
    results = sorted([r for r in results if r["timestamp"]], key=lambda x: x["timestamp"])

    # ── Stats par symbole ──
    symbols = [r["symbol"] for r in results if r["symbol"]]
    timeframes = [r["timeframe"] for r in results if r["timeframe"]]

    print("## Distribution des SYMBOLES extraits")
    print("-" * 80)
    if symbols:
        symbol_counts = Counter(symbols)
        total_symbols = sum(symbol_counts.values())
        for sym, count in symbol_counts.most_common(20):
            pct = 100 * count / total_symbols
            bar = "█" * int(pct / 2)
            print(f"{sym:15} {count:4} ({pct:5.1f}%) {bar}")
    else:
        print("⚠️  Aucun symbole extrait (peut-être pas dans les noms de dossiers)")

    print("\n## Distribution des TIMEFRAMES extraits")
    print("-" * 80)
    if timeframes:
        tf_counts = Counter(timeframes)
        total_tfs = sum(tf_counts.values())
        for tf, count in tf_counts.most_common(20):
            pct = 100 * count / total_tfs
            bar = "█" * int(pct / 2)
            print(f"{tf:15} {count:4} ({pct:5.1f}%) {bar}")
    else:
        print("⚠️  Aucun timeframe extrait")

    # ── Séquence chronologique des 30 derniers ──
    print("\n## 30 derniers runs (chronologique)")
    print("-" * 80)
    for r in results[-30:]:
        ts_str = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if r["timestamp"] else "???"
        sym = r["symbol"] or "???"
        tf = r["timeframe"] or "???"
        desc_short = r["description"][:50]
        print(f"{ts_str} | {sym:12} {tf:6} | {desc_short}")

    # ── Répétitions consécutives ──
    print("\n## RÉPÉTITIONS CONSÉCUTIVES du même symbole")
    print("-" * 80)
    if symbols:
        consecutive_runs = []
        current_symbol = None
        current_count = 0
        current_start = 0

        for idx, r in enumerate(results):
            if r["symbol"] == current_symbol:
                current_count += 1
            else:
                if current_count >= 3:  # Seuil de 3+
                    consecutive_runs.append({
                        "symbol": current_symbol,
                        "count": current_count,
                        "start_index": current_start,
                    })
                current_symbol = r["symbol"]
                current_count = 1
                current_start = idx

        if current_count >= 3:
            consecutive_runs.append({
                "symbol": current_symbol,
                "count": current_count,
                "start_index": current_start,
            })

        if consecutive_runs:
            print("🔴 Séquences de répétitions trouvées (≥3):")
            for run in sorted(consecutive_runs, key=lambda x: x["count"], reverse=True)[:10]:
                print(f"  {run['symbol']:12} → {run['count']} fois consécutives "
                      f"(runs {run['start_index']}-{run['start_index']+run['count']-1})")
        else:
            print("✅ Aucune répétition consécutive majeure (≥3) détectée.")
    else:
        print("⚠️  Pas de données pour analyser les répétitions")

    # ── Diagnostic ──
    print("\n## DIAGNOSTIC")
    print("=" * 80)

    if not symbols:
        print("❌ Aucun symbole extrait → Les métadonnées symbol/timeframe ne sont pas")
        print("   enregistrées dans session_summary.json et absentes des noms de dossiers")
    elif len(set(symbols)) == 1:
        unique_sym = list(set(symbols))[0]
        print(f"❌ UN SEUL symbole détecté ({unique_sym}) sur {len(symbols)} runs")
        print("   → auto_market_pick probablement désactivé")
        print("   → OU univers all_symbols limité à un seul token")
    elif len(set(symbols)) <= 3:
        print(f"⚠️  Très peu de symboles différents ({len(set(symbols))}) → Vérifier all_symbols")
    else:
        most_common_sym, most_common_count = Counter(symbols).most_common(1)[0]
        dominance = 100 * most_common_count / len(symbols)
        if dominance > 50:
            print(f"🔴 BIAIS MAJEUR: {most_common_sym} représente {dominance:.1f}% des runs")
        elif dominance > 33:
            print(f"⚠️  BIAIS MODÉRÉ: {most_common_sym} représente {dominance:.1f}% des runs")
        else:
            print(f"✅ Distribution acceptable (symbole max = {dominance:.1f}%)")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  ANALYSE: Sandbox Strategies Folders")
    print("=" * 80 + "\n")
    parse_sandbox_folders()
    print("\n" + "=" * 80)
    print("Analyse terminée.")
    print("=" * 80 + "\n")
