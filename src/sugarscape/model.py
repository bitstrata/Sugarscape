from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import math
import mesa
from .schedulers import ByTypeScheduler
from .agents import Sugar, Spice, Trader
from .utils import data_path, trade_volume_unique, price_gmean, get_trade

class DeathMarker(mesa.Agent):
    """Short-lived marker to visualize where a Trader died."""
    def __init__(self, unique_id, model, ttl=8):
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model)
            self.unique_id = unique_id
        self.ttl = ttl  # DO NOT set self.pos here; MultiGrid will set it

    def step(self):
        self.ttl -= 1
        if self.ttl <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)


class SugarscapeG1mt(mesa.Model):
    def __init__(self,
                 moore_movement=True,
                 width=50, height=50,
                 initial_population=200,
                 endowment_min=13, endowment_max=21,
                 metabolism_min=1, metabolism_max=2,
                 vision_min=5, vision_max=8,
                 seed=None, 
                 map_path: str | None = None,):
        # seed Mesa RNG, fallback for older Mesa
        try:
            super().__init__(seed=seed)
        except TypeError:
            super().__init__()
            if seed is not None:
                try:
                    self.random.seed(seed)
                except Exception:
                    self.random = random.Random(seed)

        self.moore_movement = moore_movement
        self.width = width
        self.height = height
        self.initial_population = initial_population
        self.endowment_min = endowment_min
        self.endowment_max = endowment_max
        self.metabolism_min = metabolism_min
        self.metabolism_max = metabolism_max
        self.vision_min = vision_min
        self.vision_max = vision_max

        # --- Load landscape ---
        default_map = Path(__file__).resolve().parents[2] / "data" / "sugar-map.txt"
        map_file = Path(map_path) if map_path else default_map
        if not map_file.exists():
            raise FileNotFoundError(f"sugar-map not found: {map_file}")

        sugar_distribution = np.genfromtxt(map_file)
        sugar_distribution = sugar_distribution+1
        # spice is mirrored horizontally, per your original code
        spice_distribution = np.flip(sugar_distribution, 1)
        spice_distribution = spice_distribution*2 - 1


        # --- Now safe to compute maxima (used by viz) ---
        self.sugar_max = float(np.max(sugar_distribution))
        self.spice_max = float(np.max(spice_distribution))
        # default viz scale for turning welfare -> green
        self.viz_welfare_cap = 100.0

        self.schedule = ByTypeScheduler(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=True)
        self._death_seq = 0

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Trader": lambda m: m.schedule.get_type_count(Trader),
                "Volume": trade_volume_unique,
                "Price": price_gmean,
            },
            agent_reporters={
                "Trade Network": lambda a: get_trade(a)
            },
        )

        # --- Place patches (note: index [y, x]) ---
        agent_id = 0
        for y in range(self.height):
            for x in range(self.width):
                max_sugar = sugar_distribution[y, x]
                if max_sugar > 0:
                    sugar = Sugar(agent_id, self, (x, y), max_sugar)
                    self.grid.place_agent(sugar, (x, y))
                    self.schedule.add(sugar)
                    agent_id += 1

                max_spice = spice_distribution[y, x]
                if max_spice > 0:
                    spice = Spice(agent_id, self, (x, y), max_spice)
                    self.grid.place_agent(spice, (x, y))
                    self.schedule.add(spice)
                    agent_id += 1

        # --- Place traders ---
        for _ in range(self.initial_population):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            sugar = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            spice = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            met_su = int(self.random.uniform(self.metabolism_min, self.metabolism_max + 1))
            met_sp = int(self.random.uniform(self.metabolism_min, self.metabolism_max + 1))
            vision = int(self.random.uniform(self.vision_min, self.vision_max))
            t = Trader(agent_id, self, (x, y),
                       moore=self.moore_movement,
                       sugar=sugar, spice=spice,
                       metabolism_sugar=met_su, metabolism_spice=met_sp,
                       vision=vision)
            self.grid.place_agent(t, (x, y))
            self.schedule.add(t); agent_id += 1

    def _randomize_traders(self):
        from .agents import Trader
        traders = list(self.schedule.agents_by_type.get(Trader, {}).values())
        self.random.shuffle(traders)
        return traders
    
    def get_sugar_amount_at(self, pos):
        for a in self.grid.get_cell_list_contents(pos):
            if isinstance(a, Sugar):
                return a.amount
        return 0.0

    def get_spice_amount_at(self, pos):
        for a in self.grid.get_cell_list_contents(pos):
            if isinstance(a, Spice):
                return a.amount
        return 0.0

    def spawn_death_marker(self, pos):
        uid = f"dead-{self.schedule.steps}-{self._death_seq}"
        self._death_seq += 1
        mark = DeathMarker(uid, self, ttl=8)
        self.grid.place_agent(mark, pos)   # sets mark.pos for us
        self.schedule.add(mark)

    def step(self):
        from .agents import Sugar, Spice, Trader
        # 1) resources grow
        for s in self.schedule.agents_by_type.get(Sugar, {}).values():
            s.step()
        for p in self.schedule.agents_by_type.get(Spice, {}).values():
            p.step()
        for mark in list(self.schedule.agents_by_type.get(DeathMarker, {}).values()):
            mark.step()

        # 2) traders move + harvest (no metabolization yet)
        traders = self._randomize_traders()
        for t in traders:
            t.prices = []
            t.trade_partners = []
            t.move()
            t.harvest()     # <- collect first

        # 3) trading rounds (now they have the harvest in hand)
        for t in self._randomize_traders():
            t.trade_with_neighbor()
        # (Optionally a second trading pass for more equilibrium)
        # for t in self._randomize_traders():
        #     t.trade_with_neighbor()

        # 4) metabolize and then deaths
        for t in list(self.schedule.agents_by_type.get(Trader, {}).values()):
            t.burn()
            t.maybe_die()

        self.schedule.steps += 1
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
