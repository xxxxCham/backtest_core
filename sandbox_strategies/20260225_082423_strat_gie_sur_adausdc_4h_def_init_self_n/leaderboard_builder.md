# Leaderboard Builder - session 20260225_082423_strat_gie_sur_adausdc_4h_def_init_self_n

Objective: Stratégie sur ADAUSDC 4h. def __init__(self, name, age): self.name = name self.age = age def __str__(self): return f"Person(name={self.name}, age={self.age})" def __repr__(self): return f"Person(name={self.name}, age={self.age})" def __eq__(self, other): if isinstance(other, Person): return self.name == other.name and self.age == other.age return False def __hash__(self): return hash((self.name, self.age)) def __lt__(self, other): if isinstance(other, Person): return self.age < other.age return NotImplemented def __le__(self, other): if isinstance(other, Person): return self.age <= other.age return NotImplemented def __gt__(self, other): if isinstance(other, Person): return self.age > other.age return NotImplemented def __ge__(self, other): if isinstance(other, Person): return self.age >= other.age return NotImplemented def __ne__(self, other): return not self.__eq__(other) def __getattr__(self, name): if name == 'is_adult': return self.age >= 18 elif name == 'is_minor': return self.age < 18 else: raise
Status: failed
Best Sharpe: 0.437
Best Continuous Score: 30.57

| Rank | Iter | Score | Sharpe | Return % | Max DD % | PF | Trades | Decision | Category |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 30.57 | 0.437 | +129.01% | -60.43% | 1.18 | 204 | continue | high_drawdown |
| 2 | 1 | -100.00 | -20.000 | -124.73% | -100.00% | 0.91 | 357 | continue | ruined |
| 3 | 3 | -100.00 | -20.000 | -186.87% | -100.00% | 0.86 | 340 | continue | ruined |
| 4 | 4 | -100.00 | -20.000 | -475.68% | -100.00% | 0.85 | 1969 | continue | ruined |
| 5 | 7 | -100.00 | -20.000 | -96.34% | -100.00% | 0.92 | 629 | continue | ruined |
| 6 | 9 | -100.00 | -20.000 | -144.51% | -100.00% | 0.62 | 426 | stop | ruined |