from __future__ import annotations
import mesa
from mesa.visualization import SolaraViz, make_space_component, make_plot_component
from sugarscape.model import SugarscapeG1mt
from sugarscape.agents import Sugar, Spice, Trader

# consistent keys for all agents; avoid alpha/edgecolors/linewidths
def agent_portrayal(a):
    style = {"marker": "o", "size": 10, "color": "gray", "zorder": 1}
    if isinstance(a, Sugar):
        style.update({"marker": "s", "size": 8, "color": "orange", "zorder": 0})
    elif isinstance(a, Spice):
        style.update({"marker": "s", "size": 8, "color": "tab:purple", "zorder": 0})
    elif isinstance(a, Trader):
        style.update({"marker": "o", "size": 16, "color": "tab:blue", "zorder": 2})
    return style

Space       = make_space_component(agent_portrayal, backend="matplotlib")
PricePlot   = make_plot_component("Price",  backend="matplotlib")
TraderPlot  = make_plot_component("Trader", backend="matplotlib")
VolumePlot  = make_plot_component("Volume", backend="matplotlib")

# default model (you can wire sliders in model_params if you want)
_page = SolaraViz(
    SugarscapeG1mt(seed=42),
    components=[Space, PricePlot, TraderPlot, VolumePlot],
    name="Sugarscape G1mt",
    play_interval=120,
)

# Solara expects a top-level component named Page
Page = _page
