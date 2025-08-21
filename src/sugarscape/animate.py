import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from .model import SugarscapeG1mt
from .agents import Sugar, Spice, Trader

def rasterize(model):
    H, W = model.height, model.width
    sugar = np.zeros((H, W)); spice = np.zeros((H, W))
    xs, ys = [], []
    for y in range(H):
        for x in range(W):
            for a in model.grid.get_cell_list_contents((x, y)):
                if isinstance(a, Sugar):
                    sugar[y, x] = a.amount
                elif isinstance(a, Spice):
                    spice[y, x] = a.amount
                elif isinstance(a, Trader):
                    xs.append(x); ys.append(y)
    return sugar, spice, np.array(xs), np.array(ys)

def main(steps=200, interval=120):
    model = SugarscapeG1mt(seed=42)
    fig, ax = plt.subplots(figsize=(6,6))
    sugar, spice, xs, ys = rasterize(model)
    im_s = ax.imshow(sugar, origin="lower", cmap="Oranges", interpolation="nearest")
    im_p = ax.imshow(spice, origin="lower", cmap="Blues", interpolation="nearest", alpha=0.35)
    sc = ax.scatter(xs, ys, s=9, c="k", alpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Sugarscape (orange=sugar, blue=spice)")

    def update(_):
        model.step()
        s, p, x, y = rasterize(model)
        im_s.set_data(s); im_p.set_data(p); sc.set_offsets(np.c_[x, y])
        return im_s, im_p, sc

    FuncAnimation(fig, update, frames=steps, interval=interval, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
