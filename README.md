# Sugarscape

Agent-based “Sugarscape” model with traders, sugar/spice resource patches, and simple trading dynamics.
Built on **Mesa 3.2.0** with both **Solara** (interactive web UI) and a **matplotlib** animation fallback.

---

## Quick start

```bash
# clone or create the repo, then:
cd /Users/markbenjamindahl/Documents/GitHub/Sugarscape

# create & activate a virtual env
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell

# install dependencies
pip install -r requirements.txt
```

Place your map file at:

```
data/sugar-map.txt
```

> This replaces the old Colab path `/content/sugar-map.txt`.

---

## Project structure

```
Sugarscape/
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ data/
│  └─ sugar-map.txt
├─ src/
│  ├─ main.py
│  └─ sugarscape/
│     ├─ __init__.py
│     ├─ model.py
│     ├─ agents.py
│     ├─ schedulers.py
│     ├─ utils.py
│     ├─ viz.py            # Solara/Mesa 3 UI
│     └─ animate.py        # matplotlib fallback
└─ tests/
   ├─ test_utils.py
   └─ test_model_smoke.py
```

---

## Run (CLI)

Run the model and save outputs to `data/`:

```bash
python src/main.py --steps 1000 --seed 123 --width 50 --height 50 --init-pop 200
```

Outputs:

* `data/model_vars.csv` — model-level time series (Trader count, Volume, Price)
* `data/agent_vars.csv` — agent-level data (e.g., trade networks)

---

## Interactive visualization (Mesa 3.x + Solara)

Mesa ≥3 uses Solara for the visualization server. Launch the UI:

```bash
solara run src/sugarscape/viz.py
```

Open the local URL Solara prints, then press **Play** to step the model.
The space view is **matplotlib**-backed; plots are live from your `DataCollector`.

> If you see `ModuleNotFoundError: No module named 'solara'`, run:
>
> ```bash
> pip install solara
> ```

---

## Fallback animation (no web server)

Prefer a simple in-window animation:

```bash
python src/sugarscape/animate.py
```

This draws sugar (orange), spice (blue heatmap), and traders (black dots) and advances the model every frame.

---

## VS Code tips

* Select the interpreter: **Command Palette → “Python: Select Interpreter” → .venv**
* Recommended extensions: **Python**, **Pylance**
* Test runner:

  ```bash
  pip install pytest
  pytest
  ```

---

## Requirements

`requirements.txt` installs:

* `mesa==3.2.0`
* `numpy`, `pandas`, `matplotlib`, `networkx`
* `solara`, `altair`, `uvicorn`, `python-multipart`, `watchfiles`

Install them with:

```bash
pip install -r requirements.txt
```

---

## Model summary

* **Agents**: `Sugar`, `Spice` (resource patches that regrow), `Trader` (moves, eats, trades)
* **Scheduler**: lightweight `ByTypeScheduler` (grouped by class, tracks steps)
* **Trading**: Cobb-Douglas welfare; geometric-mean price; repeated trade until no mutual gain
* **DataCollector**:

  * `Trader` — number of active traders
  * `Volume` — unique trade pairs per step
  * `Price` — geometric mean of prices per step

---

## Troubleshooting

* **Plotly backend error**: Mesa 3.2.0 `make_space_component` supports only `backend="matplotlib"` or `"altair"`. Plotly is not available in this version.
* **Matplotlib space errors (alpha / edgecolors)**: Keep the agent portrayal simple and consistent across agent types. In `viz.py` we return only `marker`, `size`, `color`, `zorder` for every agent.
* **Jupyter warning**: `SyntaxWarning: invalid escape sequence` from `notebookapp.py` is harmless.
* **No UI appears**: Ensure the last line of the Solara file returns the page component (already done in `viz.py` as `Page = _page`). Then run `solara run src/sugarscape/viz.py`.

---

## Development

Format, lint, or extend as you like. Minimal tests are included:

```bash
pytest
```

---

## License

MIT (or your preference). Add a `LICENSE` file if needed.
