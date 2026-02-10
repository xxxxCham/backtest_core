"""Vérifier les plages élargies"""
from strategies.bollinger_best_short_3i import BollingerBestShort3iStrategy
from strategies.bollinger_best_longe_3i import BollingerBestLonge3iStrategy

short = BollingerBestShort3iStrategy()
long = BollingerBestLonge3iStrategy()

print("="*60)
print("SHORT 3i - Plages élargies:")
print("="*60)
for k, v in short.parameter_specs.items():
    if k != 'leverage':
        print(f"  {k:15s}: {v.min_val:6} → {v.max_val:6} (default={v.default})")

print("\n" + "="*60)
print("LONG 3i - Plages élargies:")
print("="*60)
for k, v in long.parameter_specs.items():
    if k != 'leverage':
        print(f"  {k:15s}: {v.min_val:6} → {v.max_val:6} (default={v.default})")

print("\n" + "="*60)
print("Estimation combinaisons (avec granularité 0.5):")
print("="*60)

# Estimation grossière du nombre de valeurs
def estimate_values(min_val, max_val, step):
    if step:
        return int((max_val - min_val) / step) + 1
    else:
        return 4  # valeur par défaut

short_combos = 1
print("\nSHORT 3i:")
for k, v in short.parameter_specs.items():
    if k != 'leverage' and v.optimize != False:
        n = estimate_values(v.min_val, v.max_val, v.step if hasattr(v, 'step') else None)
        print(f"  {k:15s}: ~{n} valeurs")
        short_combos *= n
print(f"  TOTAL estimé : ~{short_combos:,} combinaisons (max)")

long_combos = 1
print("\nLONG 3i:")
for k, v in long.parameter_specs.items():
    if k != 'leverage' and v.optimize != False:
        n = estimate_values(v.min_val, v.max_val, v.step if hasattr(v, 'step') else None)
        print(f"  {k:15s}: ~{n} valeurs")
        long_combos *= n
print(f"  TOTAL estimé : ~{long_combos:,} combinaisons (max)")

print("\n" + "="*60)
print("Avec 90k bt/s:")
print("="*60)
print(f"  SHORT (~{short_combos:,} combos) : ~{short_combos/90000:.1f}s")
print(f"  LONG  (~{long_combos:,} combos) : ~{long_combos/90000:.1f}s")
