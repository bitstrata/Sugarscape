from __future__ import annotations
import math
from pathlib import Path
import numpy as np

def project_root() -> Path:
    # .../Sugarscape/src/sugarscape/utils.py -> parents[2] = project root
    return Path(__file__).resolve().parents[2]

def data_path(name: str) -> Path:
    return project_root() / "data" / name

def get_distance(p1, p2) -> float:
    x1, y1 = p1; x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def grid_dist(a, b, moore=False) -> int:
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) if moore else dx + dy

def flatten(list_of_lists):
    return [item for sub in list_of_lists for item in sub]

def geometric_mean(xs):
    xs = np.asarray(xs, dtype=float)
    return float(np.exp(np.log(xs).mean()))

# DataCollector helpers (wired by model)
def trade_volume_unique(m):
    from .agents import Trader
    traders = m.schedule.agents_by_type.get(Trader, {}).values()
    edges = set()
    for a in traders:
        for b_id in a.trade_partners:
            pair = tuple(sorted((a.unique_id, b_id)))
            edges.add(pair)
    return len(edges)

def price_gmean(m):
    from .agents import Trader
    traders = m.schedule.agents_by_type.get(Trader, {}).values()
    prices = [p for a in traders for p in a.prices]
    return geometric_mean(prices) if prices else float("nan")

def get_trade(agent):
    from .agents import Trader
    return agent.trade_partners if isinstance(agent, Trader) else None
