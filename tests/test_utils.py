from sugarscape.utils import grid_dist, geometric_mean
import numpy as np

def test_grid_dist():
    assert grid_dist((0,0),(1,1), moore=False) == 2
    assert grid_dist((0,0),(1,1), moore=True)  == 1

def test_geometric_mean():
    xs = [1, 4, 9]
    gm = geometric_mean(xs)
    assert np.isclose(gm, 3.0)
