from __future__ import annotations

class ByTypeScheduler:
    """Minimal scheduler grouping agents by type, with a step counter."""
    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.agents_by_type = {}  # {cls: {unique_id: agent}}

    def add(self, agent):
        d = self.agents_by_type.setdefault(type(agent), {})
        d[agent.unique_id] = agent

    def remove(self, agent):
        d = self.agents_by_type.get(type(agent))
        if d is not None:
            d.pop(agent.unique_id, None)

    def get_type_count(self, cls) -> int:
        return len(self.agents_by_type.get(cls, {}))
