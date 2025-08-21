from sugarscape import SugarscapeG1mt

def test_model_smoke():
    m = SugarscapeG1mt(seed=1)
    m.step()
    assert m.schedule.steps == 1
