"""
Module-ID: utils.gpu_monitor

Purpose: Monitoring GPU temps réel - Affichage fiable utilisation/mémoire multi-GPU.

Usage:
    python -m utils.gpu_monitor              # Mode interactif terminal
    python -m utils.gpu_monitor --web        # Mode web browser (recommandé)
    python -m utils.gpu_monitor --log        # Mode logging fichier
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

# Ajouter racine projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class GPUStats:
    """Statistiques d'un GPU à un instant T."""
    index: int
    name: str
    utilization_gpu: int      # % compute
    utilization_memory: int   # % mémoire utilisée
    memory_used_mb: int
    memory_total_mb: int
    temperature: int
    power_draw_w: float
    power_limit_w: float
    timestamp: datetime


def get_gpu_stats_nvidia_smi() -> List[GPUStats]:
    """Récupère les stats GPU via nvidia-smi (méthode la plus fiable)."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode != 0:
            return []

        stats = []
        now = datetime.now()

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue

            try:
                stats.append(GPUStats(
                    index=int(parts[0]),
                    name=parts[1],
                    utilization_gpu=int(parts[2]) if parts[2] != "[N/A]" else 0,
                    utilization_memory=int(parts[3]) if parts[3] != "[N/A]" else 0,
                    memory_used_mb=int(float(parts[4])) if parts[4] != "[N/A]" else 0,
                    memory_total_mb=int(float(parts[5])) if parts[5] != "[N/A]" else 0,
                    temperature=int(parts[6]) if parts[6] != "[N/A]" else 0,
                    power_draw_w=float(parts[7]) if parts[7] != "[N/A]" else 0,
                    power_limit_w=float(parts[8]) if parts[8] != "[N/A]" else 0,
                    timestamp=now
                ))
            except (ValueError, IndexError):
                continue

        return stats

    except Exception as e:
        print(f"Erreur nvidia-smi: {e}")
        return []


def get_gpu_processes() -> dict:
    """Récupère les processus utilisant chaque GPU."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )

        # Mapper UUID -> index
        uuid_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )

        uuid_to_idx = {}
        for line in uuid_result.stdout.strip().split("\n"):
            if "," in line:
                idx, uuid = line.split(",", 1)
                uuid_to_idx[uuid.strip()] = int(idx.strip())

        processes = {i: [] for i in uuid_to_idx.values()}

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                uuid = parts[0]
                if uuid in uuid_to_idx:
                    idx = uuid_to_idx[uuid]
                    processes[idx].append({
                        "pid": parts[1],
                        "name": parts[2],
                        "memory_mb": parts[3]
                    })

        return processes

    except Exception:
        return {}


def print_gpu_bar(value: int, max_val: int = 100, width: int = 30, label: str = "") -> str:
    """Génère une barre de progression ASCII."""
    filled = int(width * value / max_val)
    bar = "█" * filled + "░" * (width - filled)

    # Couleur selon utilisation
    if value < 30:
        color = "\033[92m"  # Vert
    elif value < 70:
        color = "\033[93m"  # Jaune
    else:
        color = "\033[91m"  # Rouge

    reset = "\033[0m"
    return f"{label}{color}{bar}{reset} {value:3d}%"


def clear_screen():
    """Efface l'écran terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')


def run_terminal_monitor(interval: float = 1.0):
    """Mode monitoring terminal interactif."""
    print("\033[?25l")  # Cacher curseur

    try:
        while True:
            clear_screen()
            stats = get_gpu_stats_nvidia_smi()
            processes = get_gpu_processes()

            print("=" * 70)
            print(f"  GPU MONITOR - {datetime.now().strftime('%H:%M:%S')}  (Ctrl+C pour quitter)")
            print("=" * 70)

            if not stats:
                print("\n  ⚠️  Aucun GPU NVIDIA détecté ou nvidia-smi indisponible")
                time.sleep(interval)
                continue

            for gpu in stats:
                mem_pct = int(100 * gpu.memory_used_mb / gpu.memory_total_mb) if gpu.memory_total_mb > 0 else 0

                print(f"\n  GPU {gpu.index}: {gpu.name}")
                print(f"  {'─' * 50}")
                print(f"  {print_gpu_bar(gpu.utilization_gpu, label='  Compute: ')}")
                print(f"  {print_gpu_bar(mem_pct, label='  Mémoire: ')}  ({gpu.memory_used_mb:,} / {gpu.memory_total_mb:,} MB)")
                print(f"  🌡️  Temp: {gpu.temperature}°C   ⚡ Power: {gpu.power_draw_w:.0f}W / {gpu.power_limit_w:.0f}W")

                # Processus sur ce GPU
                if gpu.index in processes and processes[gpu.index]:
                    print(f"  📋 Processus:")
                    for proc in processes[gpu.index][:5]:  # Max 5
                        print(f"      • {proc['name'][:30]:30s}  {proc['memory_mb']:>6s} MB")

            print("\n" + "=" * 70)
            print("  💡 Note: Ces valeurs viennent de nvidia-smi (CUDA), pas de Windows")
            print("=" * 70)

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\033[?25h")  # Réafficher curseur
        print("\n\nMonitoring arrêté.")


def run_web_monitor(port: int = 8765):
    """Mode monitoring web avec graphiques temps réel."""
    try:
        from http.server import HTTPServer, SimpleHTTPRequestHandler
        import json
        import threading
    except ImportError as e:
        print(f"Erreur import: {e}")
        return

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>GPU Monitor - Backtest Core</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #00d4ff;
        }
        .gpu-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }
        .gpu-card {
            background: #16213e;
            border-radius: 15px;
            padding: 20px;
            min-width: 350px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .gpu-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #00d4ff;
        }
        .metric {
            margin: 10px 0;
        }
        .metric-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .progress-bar {
            height: 25px;
            background: #0f3460;
            border-radius: 12px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s ease, background 0.3s ease;
            border-radius: 12px;
        }
        .fill-green { background: linear-gradient(90deg, #00c853, #69f0ae); }
        .fill-yellow { background: linear-gradient(90deg, #ffc107, #ffeb3b); }
        .fill-red { background: linear-gradient(90deg, #ff5252, #ff8a80); }
        .stats-row {
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #0f3460;
        }
        .stat-item {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.8em;
            color: #888;
        }
        .chart-container {
            margin-top: 15px;
            height: 100px;
            background: #0f3460;
            border-radius: 10px;
            padding: 10px;
            position: relative;
        }
        canvas { width: 100% !important; height: 100% !important; }
        .timestamp {
            text-align: center;
            margin-top: 20px;
            color: #666;
        }
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 10px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <h1>🖥️ GPU Monitor - Backtest Core</h1>
    <div id="gpus" class="gpu-container"></div>
    <div class="timestamp" id="timestamp"></div>
    <div class="legend">
        <div class="legend-item"><div class="legend-color fill-green"></div> &lt;30%</div>
        <div class="legend-item"><div class="legend-color fill-yellow"></div> 30-70%</div>
        <div class="legend-item"><div class="legend-color fill-red"></div> &gt;70%</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const history = {};
        const charts = {};
        const MAX_HISTORY = 60;

        function getColorClass(value) {
            if (value < 30) return 'fill-green';
            if (value < 70) return 'fill-yellow';
            return 'fill-red';
        }

        function updateGPU(gpu) {
            const id = `gpu-${gpu.index}`;
            let card = document.getElementById(id);

            if (!card) {
                card = document.createElement('div');
                card.id = id;
                card.className = 'gpu-card';
                document.getElementById('gpus').appendChild(card);
            }

            const memPct = Math.round(100 * gpu.memory_used_mb / gpu.memory_total_mb);

            card.innerHTML = `
                <div class="gpu-name">GPU ${gpu.index}: ${gpu.name}</div>
                <div class="metric">
                    <div class="metric-label">
                        <span>⚡ Compute</span>
                        <span>${gpu.utilization_gpu}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${getColorClass(gpu.utilization_gpu)}"
                             style="width: ${gpu.utilization_gpu}%"></div>
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">
                        <span>💾 Mémoire</span>
                        <span>${memPct}% (${gpu.memory_used_mb.toLocaleString()} MB)</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${getColorClass(memPct)}"
                             style="width: ${memPct}%"></div>
                    </div>
                </div>
                <div class="stats-row">
                    <div class="stat-item">
                        <div class="stat-value">${gpu.temperature}°C</div>
                        <div class="stat-label">🌡️ Temp</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${gpu.power_draw_w.toFixed(0)}W</div>
                        <div class="stat-label">⚡ Power</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">${gpu.memory_total_mb.toLocaleString()}</div>
                        <div class="stat-label">💾 Total MB</div>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="chart-${gpu.index}"></canvas>
                </div>
            `;

            // Update history
            if (!history[gpu.index]) {
                history[gpu.index] = { compute: [], memory: [], labels: [] };
            }
            const h = history[gpu.index];
            h.compute.push(gpu.utilization_gpu);
            h.memory.push(memPct);
            h.labels.push('');

            if (h.compute.length > MAX_HISTORY) {
                h.compute.shift();
                h.memory.shift();
                h.labels.shift();
            }

            // Update chart
            const ctx = document.getElementById(`chart-${gpu.index}`);
            if (ctx) {
                if (!charts[gpu.index]) {
                    charts[gpu.index] = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: h.labels,
                            datasets: [
                                {
                                    label: 'Compute %',
                                    data: h.compute,
                                    borderColor: '#00d4ff',
                                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                                    fill: true,
                                    tension: 0.3
                                },
                                {
                                    label: 'Memory %',
                                    data: h.memory,
                                    borderColor: '#ff6b6b',
                                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                                    fill: true,
                                    tension: 0.3
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: { legend: { display: false } },
                            scales: {
                                x: { display: false },
                                y: { min: 0, max: 100, display: false }
                            },
                            animation: { duration: 0 }
                        }
                    });
                } else {
                    charts[gpu.index].data.datasets[0].data = h.compute;
                    charts[gpu.index].data.datasets[1].data = h.memory;
                    charts[gpu.index].data.labels = h.labels;
                    charts[gpu.index].update('none');
                }
            }
        }

        async function fetchStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();

                data.forEach(gpu => updateGPU(gpu));
                document.getElementById('timestamp').textContent =
                    `Dernière mise à jour: ${new Date().toLocaleTimeString()} (nvidia-smi)`;
            } catch (e) {
                console.error('Erreur fetch:', e);
            }
        }

        // Refresh toutes les secondes
        setInterval(fetchStats, 1000);
        fetchStats();
    </script>
</body>
</html>"""

    class MonitorHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(html_content.encode())
            elif self.path == '/api/stats':
                stats = get_gpu_stats_nvidia_smi()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                data = []
                for s in stats:
                    data.append({
                        'index': s.index,
                        'name': s.name,
                        'utilization_gpu': s.utilization_gpu,
                        'utilization_memory': s.utilization_memory,
                        'memory_used_mb': s.memory_used_mb,
                        'memory_total_mb': s.memory_total_mb,
                        'temperature': s.temperature,
                        'power_draw_w': s.power_draw_w,
                        'power_limit_w': s.power_limit_w
                    })

                self.wfile.write(json.dumps(data).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # Silence logs

    print(f"\n🖥️  GPU Monitor Web Server")
    print(f"   Ouvrez: http://localhost:{port}")
    print(f"   Ctrl+C pour arrêter\n")

    # Ouvrir navigateur automatiquement
    import webbrowser
    webbrowser.open(f"http://localhost:{port}")

    server = HTTPServer(('localhost', port), MonitorHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServeur arrêté.")


def run_log_mode(interval: float = 1.0, output_file: str = "gpu_stats.csv"):
    """Mode logging CSV pour analyse ultérieure."""
    import csv

    print(f"📝 Logging GPU stats vers {output_file} (Ctrl+C pour arrêter)")

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'timestamp', 'gpu_index', 'gpu_name', 'utilization_gpu',
            'utilization_memory', 'memory_used_mb', 'memory_total_mb',
            'temperature', 'power_draw_w'
        ])

        try:
            while True:
                stats = get_gpu_stats_nvidia_smi()
                for gpu in stats:
                    writer.writerow([
                        gpu.timestamp.isoformat(),
                        gpu.index,
                        gpu.name,
                        gpu.utilization_gpu,
                        gpu.utilization_memory,
                        gpu.memory_used_mb,
                        gpu.memory_total_mb,
                        gpu.temperature,
                        gpu.power_draw_w
                    ])
                f.flush()
                print(f"  {datetime.now().strftime('%H:%M:%S')} - {len(stats)} GPU(s) logged")
                time.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n✅ Log sauvegardé: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="GPU Monitor - Backtest Core")
    parser.add_argument('--web', action='store_true', help="Mode web avec graphiques (recommandé)")
    parser.add_argument('--log', action='store_true', help="Mode logging CSV")
    parser.add_argument('--interval', type=float, default=1.0, help="Intervalle refresh (secondes)")
    parser.add_argument('--port', type=int, default=8765, help="Port serveur web")
    parser.add_argument('--output', type=str, default="gpu_stats.csv", help="Fichier log CSV")

    args = parser.parse_args()

    if args.web:
        run_web_monitor(port=args.port)
    elif args.log:
        run_log_mode(interval=args.interval, output_file=args.output)
    else:
        run_terminal_monitor(interval=args.interval)


if __name__ == "__main__":
    main()