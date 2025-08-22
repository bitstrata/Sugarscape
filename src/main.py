import argparse
from pathlib import Path
import pandas as pd
from sugarscape import SugarscapeG1mt

def parse_args():
    p = argparse.ArgumentParser(description="Run Sugarscape")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--width", type=int, default=50)
    p.add_argument("--height", type=int, default=50)
    p.add_argument("--init-pop", type=int, default=200)
    p.add_argument("--sugar-noise", type=float, default=0.5)
    p.add_argument("--spice-noise", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    model = SugarscapeG1mt(
        width=args.width, height=args.height,
        initial_population=args.init_pop, seed=args.seed,
        sugar_noise_sigma=args.sugar_noise,
        spice_noise_sigma=args.spice_noise,
    )
    model.run_model(step_count=args.steps)
    mv = model.datacollector.get_model_vars_dataframe()
    av = model.datacollector.get_agent_vars_dataframe()
    out_dir = Path("data")
    mv.to_csv(out_dir / "model_vars.csv", index=True)
    av.to_csv(out_dir / "agent_vars.csv", index=True)
    print(f"Saved to {out_dir}/model_vars.csv and agent_vars.csv")

if __name__ == "__main__":
    main()
