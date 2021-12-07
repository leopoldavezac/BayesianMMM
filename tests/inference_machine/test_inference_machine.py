from numpy.core.defchararray import array
import pytest
import numpy as np

from bayesian_mmm.inference_machine.inference_machine import InferenceMachine


SPENDS = np.array([
    [10, 20],
    [0, 8],
    [1, 30],
    [5, 40]
])
CTRL_VARS = np.array([
    [2, 4],
    [5, 2],
    [6, 4],
    [7, 2]
])

DELAYS = [1.5, 1]
RETAIN_RATES = [0.2, 0.9]
ECS = [0.2, 1]
SLOPES = [1, 4]
HALF_SATURATIONS = [2, 3]
BETA_MEDIAS = [0.2, 0.4]
GAMMA_CTRLS = [0.5, 1]
TAU = 2


@pytest.mark.parametrize(
    "carryover_transfo_nm,diminushing_returns_transfo_nm,with_ctrl_vars,max_lag,expected_y",
    [
        ("adstock","hill",True, 2, np.array([7.531529, 7.095915, 9.432255, 8.070454])),
        ("adstock","hill",False, 3, np.array([2.498776, 2.592115, 2.592187, 2.549253])),
        ("adstock","reach",True, 2, np.array([7.473342, 7.100000, 9.407689, 8.063805])),
        ("adstock","reach",False, 3, np.array([2.438721, 2.599978, 2.599979, 2.505725])),
        ("geo_decay","hill",True, 2, np.array([7.595280, 7.078560, 9.561288, 8.091176])),
        ("geo_decay","hill",False, 3, np.array([2.595025, 2.577889, 2.569900, 2.590895])),
        ("geo_decay","reach",True, 2, np.array([7.600000, 7.086222, 9.536452, 8.099931])),
        ("geo_decay","reach",False, 3, np.array([2.600000, 2.584718, 2.562138, 2.599909]))
    ]
)
def test_predict(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars,
    max_lag,
    expected_y
    ):
    param_nm_to_val = {
        "beta_medias":BETA_MEDIAS,
        "tau":TAU
    }
    
    if carryover_transfo_nm == "adstock":
        param_nm_to_val.update({
            "retain_rate":RETAIN_RATES,
            "delay":DELAYS
        })
    elif carryover_transfo_nm == "geo_decay":
        param_nm_to_val["retain_rate"] = RETAIN_RATES

    if diminushing_returns_transfo_nm == "hill":
        param_nm_to_val.update({
            "ec":ECS,
            "slope":SLOPES
        })
    elif diminushing_returns_transfo_nm == "reach":
        param_nm_to_val["half_saturation"] = HALF_SATURATIONS

    if with_ctrl_vars:
        param_nm_to_val["gamma_ctrl"] = GAMMA_CTRLS
        X_ctrl = CTRL_VARS
    else:
        X_ctrl = None

    inference_machine = InferenceMachine(param_nm_to_val, max_lag)
    obtained_y = inference_machine.predict(SPENDS, X_ctrl)

    print(obtained_y, "\n\n", expected_y)

    print("np.array([%f, %f, %f, %f])" % (obtained_y[0], obtained_y[1], obtained_y[2], obtained_y[3]))

    assert np.allclose(obtained_y, expected_y) #allclose to ignore precision differences
