# src/sugarscape/viz.py
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from sugarscape.agents import Sugar, Spice, Trader
from sugarscape.model import SugarscapeG1mt, DeathMarker

def agent_portrayal(a):
    if isinstance(a, Sugar):
        m = a.model
        r = 0 if m.sugar_max <= 0 else min(1.0, a.amount / m.sugar_max)
        b = 0 if m.spice_max <= 0 else min(1.0, m.get_spice_amount_at(a.pos) / m.spice_max)
        return {"marker":"s", "size":120, "color":(r, 0.05, b), "zorder":0}
    if isinstance(a, Spice):
        return {"marker":"s", "size":0, "color":(0,0,0), "zorder":0}
    if isinstance(a, Trader):
        w = a.calculate_welfare(a.sugar, a.spice)
        cap = getattr(a.model, "viz_welfare_cap", 100.0)
        g = 0.0 if cap <= 0 else max(0.0, min(1.0, w/cap))
        return {"marker":"o", "size":16, "color":(0.1, g, 0.1), "zorder":3}
    if isinstance(a, DeathMarker):
        return {"marker":"x", "size":12, "color":(0.6,0.6,0.6), "zorder":2}
    return {"marker":"o", "size":8, "color":"gray", "zorder":1}

Space      = make_space_component(agent_portrayal, backend="matplotlib")
PricePlot  = make_plot_component("Price", backend="matplotlib")
TraderPlot = make_plot_component("Trader", backend="matplotlib")
VolumePlot = make_plot_component("Volume", backend="matplotlib")

page = SolaraViz(
    SugarscapeG1mt(
        seed=42,
        sugar_noise_sigma=0.3,
        spice_noise_sigma=0.5,
        ),
    components=[Space, PricePlot, TraderPlot, VolumePlot],
    name="Sugarscape G1mt",
    play_interval=200,
)