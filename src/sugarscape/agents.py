from __future__ import annotations
import math
import numpy as np
import mesa
from .utils import grid_dist

class Sugar(mesa.Agent):
    def __init__(self, unique_id, model, pos, max_sugar):
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model); self.unique_id = unique_id
        self.model = model
        self.amount = max_sugar
        self.max_sugar = max_sugar

    def step(self):
        self.amount = min(self.max_sugar, self.amount + 1)

class Spice(mesa.Agent):
    def __init__(self, unique_id, model, pos, max_spice):
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model); self.unique_id = unique_id
        self.model = model
        self.amount = max_spice
        self.max_spice = max_spice

    def step(self):
        self.amount = min(self.max_spice, self.amount + 1)

class Trader(mesa.Agent):
    def __init__(self, unique_id, model, pos, moore=False, sugar=0, spice=0,
                 metabolism_sugar=0, metabolism_spice=0, vision=0):
        try:
            super().__init__(unique_id, model)
        except TypeError:
            super().__init__(model); self.unique_id = unique_id
        self.model = model
        self.moore = moore
        self.sugar = sugar
        self.spice = spice
        self.metabolism_sugar = metabolism_sugar
        self.metabolism_spice = metabolism_spice
        self.vision = vision
        self.prices = []
        self.trade_partners = []

    # --- cell queries ---
    def get_sugar(self, pos):
        for a in self.model.grid.get_cell_list_contents(pos):
            if isinstance(a, Sugar): return a
        return None

    def get_spice(self, pos):
        for a in self.model.grid.get_cell_list_contents(pos):
            if isinstance(a, Spice): return a
        return None

    def get_trader(self, pos):
        for a in self.model.grid.get_cell_list_contents(pos):
            if isinstance(a, Trader): return a
        return None

    def is_occupied_by_other_trader(self, pos):
        if pos == self.pos: return False
        for a in self.model.grid.get_cell_list_contents(pos):
            if isinstance(a, Trader): return True
        return False

    # --- quantities at pos ---
    def get_sugar_amount(self, pos): 
        p = self.get_sugar(pos);  return p.amount if p else 0
    def get_spice_amount(self, pos): 
        p = self.get_spice(pos);  return p.amount if p else 0

    # --- economics ---
    def calculate_welfare(self, sugar, spice):
        m_total = self.metabolism_sugar + self.metabolism_spice
        if m_total <= 0: return 0.0
        a_su = self.metabolism_sugar / m_total
        a_sp = self.metabolism_spice / m_total
        sugar = max(0.0, float(sugar))
        spice = max(0.0, float(spice))
        return (sugar ** a_su) * (spice ** a_sp)

    def calculate_MRS(self):
        eps = 1e-9
        mr_su = max(self.sugar, eps) / max(self.metabolism_sugar, eps)
        mr_sp = max(self.spice, eps) / max(self.metabolism_spice, eps)
        return mr_sp / mr_su

    def calculate_amount_exchanged(self, price):
        eps = 1e-9
        if not math.isfinite(price) or price <= eps: return None
        if price >= 1:
            return 1, max(1, int(round(price)))
        else:
            return max(1, int(round(1.0 / price))), 1

    def exchange_resource(self, nbor, sugar_exchanged, spice_exchanged):
        self.sugar += sugar_exchanged
        self.spice -= spice_exchanged
        nbor.sugar -= sugar_exchanged
        nbor.spice += spice_exchanged

    def maybe_sell_resource(self, nbor, price, welfare_self, welfare_nbor, selling, min_gain=1e-12):
        amt = self.calculate_amount_exchanged(price)
        if amt is None: return False
        sugar_ex, spice_ex = amt

        if selling == "sugar":
            dss, dsp = -sugar_ex, +spice_ex
            dns, dnp = +sugar_ex, -spice_ex
        else:  # "spice"
            dss, dsp = +sugar_ex, -spice_ex
            dns, dnp = -sugar_ex, +spice_ex

        self_su = self.sugar + dss; self_sp = self.spice + dsp
        nbor_su = nbor.sugar + dns; nbor_sp = nbor.spice + dnp
        if min(self_su, self_sp, nbor_su, nbor_sp) < 0: return False

        ws_new = self.calculate_welfare(self_su, self_sp)
        wn_new = nbor.calculate_welfare(nbor_su, nbor_sp)
        if (ws_new - welfare_self) < min_gain or (wn_new - welfare_nbor) < min_gain:
            return False

        def mrs_with(agent, su, sp):
            return ((sp / max(1e-9, agent.metabolism_spice)) /
                    (su / max(1e-9, agent.metabolism_sugar)))

        mrs_self_b = self.calculate_MRS()
        mrs_nbor_b = nbor.calculate_MRS()
        mrs_self_a = mrs_with(self, self_su, self_sp)
        mrs_nbor_a = mrs_with(nbor, nbor_su, nbor_sp)

        if selling == "spice":
            if not (mrs_self_b > mrs_nbor_b and mrs_self_a >= mrs_nbor_a): return False
        else:
            if not (mrs_self_b < mrs_nbor_b and mrs_self_a <= mrs_nbor_a): return False

        self.exchange_resource(nbor, sugar_ex, spice_ex)
        return True

    def trade(self, nbor, max_rounds=64, min_gain=1e-12):
        rounds = 0
        while rounds < max_rounds:
            if (self.sugar <= 0 and self.spice <= 0) or (nbor.sugar <= 0 and nbor.spice <= 0):
                break
            m_self = self.calculate_MRS()
            m_nbor = nbor.calculate_MRS()
            if not (math.isfinite(m_self) and math.isfinite(m_nbor)): break
            if math.isclose(m_self, m_nbor, rel_tol=1e-9, abs_tol=1e-12): break
            price = math.sqrt(m_self * m_nbor)
            if not (math.isfinite(price) and price > 0): break
            w_self = self.calculate_welfare(self.sugar, self.spice)
            w_nbor = nbor.calculate_welfare(nbor.sugar, nbor.spice)

            if m_self > m_nbor:
                sold = self.maybe_sell_resource(nbor, price, w_self, w_nbor, "spice", min_gain)
            else:
                sold = self.maybe_sell_resource(nbor, price, w_self, w_nbor, "sugar", min_gain)
            if not sold: break

            self.prices.append(price)
            self.trade_partners.append(nbor.unique_id)
            rounds += 1

    # --- main movement/eat/die/trade-with-neighbor ---
    def move(self):
        neighbors = [i for i in self.model.grid.get_neighborhood(
            self.pos, self.moore, True, self.vision
        ) if not self.is_occupied_by_other_trader(i)]
        if not neighbors: return
        welfares = [
            self.calculate_welfare(
                self.sugar + self.get_sugar_amount(pos),
                self.spice + self.get_spice_amount(pos),
            )
            for pos in neighbors
        ]
        assert all(isinstance(w, (int, float, np.floating)) and np.isfinite(w) for w in welfares)
        max_w = max(welfares)
        idxs = [i for i, w in enumerate(welfares) if math.isclose(w, max_w)]
        cands = [neighbors[i] for i in idxs]
        min_d = min(grid_dist(self.pos, p, self.moore) for p in cands)
        finals = [p for p in cands if math.isclose(grid_dist(self.pos, p, self.moore), min_d, rel_tol=1e-3)]
        self.model.random.shuffle(finals)
        self.model.grid.move_agent(self, finals[0])

    def eat(self):
        p = self.get_sugar(self.pos)
        if p: self.sugar += p.amount; p.amount = 0
        self.sugar = max(0, self.sugar - self.metabolism_sugar)
        q = self.get_spice(self.pos)
        if q: self.spice += q.amount; q.amount = 0
        self.spice = max(0, self.spice - self.metabolism_spice)

    def maybe_die(self):
        if self.sugar <= 0 and self.spice <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)

    def trade_with_neighbor(self):
        nbors = [
            self.get_trader(pos) for pos in self.model.grid.get_neighborhood(
                self.pos, self.moore, False, self.vision)
            if self.is_occupied_by_other_trader(pos)
        ]
        for a in nbors:
            if a: self.trade(a)
