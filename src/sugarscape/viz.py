import math
import solara as sol
from mesa.visualization import SolaraViz, make_space_component, make_plot_component

from .model import SugarscapeG1mt
from .agents import Trader, Sugar, Spice, DeathMarker  # make sure these import paths match your layout


# ---------- Agent portrayal ---------------------------------------------------
def agent_portrayal(agent):
    """
    Color scheme:
      - Background cells are drawn by the Sugar agent in a cell:
          color = (sugar_norm, 0.05, spice_norm)
      - Spice agent draws nothing (we avoid double-drawing).
      - Trader color's green channel encodes welfare; size encodes total stock.
      - DeathMarker is a small gray 'x' that fades out by TTL.
    """
    # base style keys accepted by Mesa's matplotlib space
    if isinstance(agent, Sugar):
        m = agent.model
        # combine sugar (R) and spice (B) at this cell
        sugar = float(agent.amount)
        spice = float(m.get_spice_amount_at(agent.pos))
        r = 0.0 if m.sugar_max <= 0 else max(0.0, min(1.0, sugar / m.sugar_max))
        b = 0.0 if m.spice_max <= 0 else max(0.0, min(1.0, spice / m.spice_max))
        return {
            "marker": "s",
            "size": 120,           # square that nicely fills cell in most themes
            "color": (r, 0.05, b), # (R, G, B)
            "zorder": 0,
        }

    if isinstance(agent, Spice):
        # we already encoded spice via the Sugar tile; skip drawing
        return {"marker": "s", "size": 0, "color": (0, 0, 0), "zorder": 0}

    if isinstance(agent, Trader):
        m = agent.model
        # welfare drives green channel; clamp by viz_welfare_cap
        welfare = agent.calculate_welfare(agent.sugar, agent.spice)
        cap = getattr(m, "viz_welfare_cap", 100.0)
        g = 0.0 if cap <= 0 else max(0.0, min(1.0, welfare / cap))

        # size scales with "wealth" (sugar+spice), but cap so dots don't explode
        wealth = float(agent.sugar + agent.spice)
        size = 18 + 12 * (1 - math.exp(-wealth / 40.0))  # smooth saturating growth

        # near-starvation warning: yellowish if about to starve next tick
        will_starve = (agent.sugar + agent.spice) <= (agent.metabolism_sugar + agent.metabolism_spice)
        color = (0.85, 0.75, 0.1) if will_starve else (0.1, g, 0.1)

        return {
            "marker": "o",
            "size": size,
            "color": color,
            "zorder": 3,
        }

    if isinstance(agent, DeathMarker):
        # fade by TTL (alpha-ish effect by mixing toward background)
        # keep alpha constant to avoid mpl array-alpha bugs; fade via color
        t = max(0, min(1, agent.ttl / 8))
        gray = 0.25 + 0.5 * t
        return {
            "marker": "x",
            "size": 14,
            "color": (gray, gray, gray),
            "zorder": 2,
        }

    # default fallback
    return {"marker": "o", "size": 8, "color": "gray", "zorder": 1}


# ---------- Optional small legend: red/blue resource grid ---------------------
def _post_process(ax):
    """Draw a tiny 2D color legend (sugar=R, spice=B) in the lower-left corner."""
    try:
        # clear previous inset if we added one
        for child in list(ax.callbacks.callbacks.get('xlim_changed', {}).values()):
            pass  # nothing needed; keeping this simple/robust
        # build a small RB gradient legend
        import numpy as np
        w, h = 20, 20
        r = np.linspace(0, 1, w)
        b = np.linspace(0, 1, h)
        R, B = np.meshgrid(r, b)
        RGB = np.dstack([R, 0.05 * np.ones_like(R), B])

        inset = ax.inset_axes([0.01, 0.03, 0.18, 0.18])  # x, y, w, h in axes fraction
        inset.imshow(RGB, origin="lower", interpolation="nearest")
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title("Cell color = (Sugar=R, Spice=B)", fontsize=8)
    except Exception:
        # never let legend drawing crash the viz
        pass


# ---------- Compose the page --------------------------------------------------
# Build the space component with our portrayal and legend
Space = make_space_component(agent_portrayal, backend="matplotlib", post_process=_post_process)
PricePlot  = make_plot_component("Price", backend="matplotlib")
TraderPlot = make_plot_component("Trader", backend="matplotlib")
VolumePlot = make_plot_component("Volume", backend="matplotlib")


# Wrap the Space in a small control panel so you can adjust the welfare scale live
@sol.component
def SpaceWithControls(model: SugarscapeG1mt):
    cap, set_cap = sol.use_state(100.0)
    # sync to model so the portrayal can read it
    model.viz_welfare_cap = cap

    with sol.HBox(gap="1rem"):
        # controls
        with sol.VBox(gap="0.5rem", style={"minWidth": "260px"}):
            sol.Markdown("### Display")
            sol.SliderFloat("Welfare → Green cap", value=cap, on_value=set_cap, min=10.0, max=300.0, step=5.0)
            sol.Markdown(
                "- **Red**: sugar amount (per cell)\n"
                "- **Blue**: spice amount (per cell)\n"
                "- **Green**: trader welfare (higher = greener)\n"
                "- **Yellow dots**: traders likely to starve next step\n"
                "- **Gray ×**: recent deaths"
            )
        # the actual grid
        Space(model)


# A handy model factory for the page
def _make_model():
    return SugarscapeG1mt(width=50, height=50, initial_population=200, seed=42)


# The Solara page
page = SolaraViz(
    _make_model(),                         # pass an instance (or the class)
    components=[SpaceWithControls, PricePlot, TraderPlot, VolumePlot],
    name="Sugarscape G1mt",
    play_interval=120,
)
