from __future__ import annotations
import random
import numpy as np
import mesa
from .schedulers import ByTypeScheduler
from .agents import Sugar, Spice, Trader
from .utils import data_path, trade_volume_unique, price_gmean, get_trade

class SugarscapeG1mt(mesa.Model):
    def __init__(self,
                 width=50, height=50,
                 initial_population=200,
                 endowment_min=25, endowment_max=50,
                 metabolism_min=1, metabolism_max=5,
                 vision_min=1, vision_max=5,
                 seed=None,
                 map_path=None):
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

        self.width = width
        self.height = height
        self.initial_population = initial_population
        self.endowment_min = endowment_min
        self.endowment_max = endowment_max
        self.metabolism_min = metabolism_min
        self.metabolism_max = metabolism_max
        self.vision_min = vision_min
        self.vision_max = vision_max

        self.schedule = ByTypeScheduler(self)
        self.grid = mesa.space.MultiGrid(self.width, self.height, torus=False)

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

        # load landscape
        mp = data_path("sugar-map.txt") if map_path is None else map_path
        sugar_distribution = np.genfromtxt(str(mp))
        spice_distribution = np.flip(sugar_distribution, 1)

        agent_id = 0
        for y in range(self.height):
            for x in range(self.width):
                max_s = sugar_distribution[y, x]
                if max_s > 0:
                    s = Sugar(agent_id, self, (x, y), max_s)
                    self.grid.place_agent(s, (x, y))
                    self.schedule.add(s); agent_id += 1
                max_p = spice_distribution[y, x]
                if max_p > 0:
                    p = Spice(agent_id, self, (x, y), max_p)
                    self.grid.place_agent(p, (x, y))
                    self.schedule.add(p); agent_id += 1

        for _ in range(self.initial_population):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            sugar = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            spice = int(self.random.uniform(self.endowment_min, self.endowment_max + 1))
            met_su = int(self.random.uniform(self.metabolism_min, self.metabolism_max + 1))
            met_sp = int(self.random.uniform(self.metabolism_min, self.metabolism_max + 1))
            vision = int(self.random.uniform(self.vision_min, self.vision_max))
            t = Trader(agent_id, self, (x, y),
                       moore=False,
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

    def step(self):
        from .agents import Sugar, Spice, Trader
        # resources grow
        for s in self.schedule.agents_by_type.get(Sugar, {}).values():
            s.step()
        for p in self.schedule.agents_by_type.get(Spice, {}).values():
            p.step()
        # traders move/eat/die
        for t in self._randomize_traders():
            t.prices = []
            t.trade_partners = []
            t.move(); t.eat(); t.maybe_die()
        # traders trade
        for t in self._randomize_traders():
            t.trade_with_neighbor()

        self.schedule.steps += 1
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
