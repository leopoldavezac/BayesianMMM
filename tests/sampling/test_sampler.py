from _pytest.mark import param
import pytest
import numpy as np

from bayesian_mmm.sampling.stan_model_generator import StanModelGenerator
from bayesian_mmm.sampling.sampler import Sampler
from bayesian_mmm.sampling.stan_model_wrapper import StanModelWrapper

MAX_LAG = 4
SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
LAGGED_SPENDS = np.array([
    [[10,  0, 0, 0], [20,  0, 0, 0]],
    [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
    [[ 1,  0, 10, 0], [30,  8, 20, 0]],
    [[ 5,  1, 0, 10], [40, 30, 8, 20]]
])
CTRL_VARS = np.array([
    [2, 4],
    [5, 2],
    [6, 4],
    [7, 2]
])
REVENUE = np.array([1, 2, 3, 4])
N = 4
NUM_MEDIA = 2
NUM_CTRL = 2

STAN_MODEL = StanModelWrapper



@pytest.mark.parametrize(
    "ctrl_vars", [CTRL_VARS, None]
)
def test_create_sampler_input(ctrl_vars):

    if type(ctrl_vars) == np.ndarray:

        expected_args = {
            "N":N,
            "Y":REVENUE,
            "max_lag":MAX_LAG,
            "num_media":NUM_MEDIA,
            "X_media":LAGGED_SPENDS,
            "num_ctrl":NUM_CTRL,
            "X_ctrl":CTRL_VARS
        }
    else:
        expected_args = {
            "N":N,
            "Y":REVENUE,
            "max_lag":MAX_LAG,
            "num_media":NUM_MEDIA,
            "X_media":LAGGED_SPENDS
        }

    sampler = Sampler(STAN_MODEL, MAX_LAG)
    sampler.create_stan_input(
        SPENDS, ctrl_vars, REVENUE
    )
    obtained_args = sampler._Sampler__args

    expected_args_keys = list(expected_args.keys())
    expected_args_keys.sort()
    obtained_args_keys = list(obtained_args.keys())
    obtained_args_keys.sort()

    assert obtained_args_keys == expected_args_keys

    for key, val in expected_args.items():
        if type(val) == np.ndarray:
            assert (val == obtained_args[key]).all()
        else:
            assert val == obtained_args[key]


# slow to run (stan compilation + sampling)
@pytest.mark.parametrize(
    "carryover_transfo_nm,diminushing_returns_transfo_nm,with_ctrl_vars",
    [
        ("adstock","hill",True),
        ("adstock","hill",False),
        ("adstock","reach",True),
        ("adstock","reach",False),
        ("geo_decay","hill",True),
        ("geo_decay","hill",False),
        ("geo_decay","reach",True),
        ("geo_decay","reach",False)
    ]
)
def test_run_sampling(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars
):


    CARRYOVER_TRANSFO_NM_TO_PARAM_NM = {
        "geo_decay":["retain_rate"],
        "adstock":["retain_rate", "delay"]
    }

    DIMINUSHING_RETURNS_TRANSFO_NM_TO_PARAM_NM = {
        "hill":["ec", "slope"],
        "reach":["half_saturation"]
    }

    WITH_CTRL_VARS_TO_PARAM_NM = {
        True:["gamma_ctrl"],
        False:[]
    }

    stan_model_generator = StanModelGenerator(
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        with_ctrl_vars
    )
    stan_model_generator.create_model()
    stan_model = stan_model_generator.get_model()

    sampler = Sampler(stan_model, MAX_LAG)

    if with_ctrl_vars:
        ctrl_vars = CTRL_VARS
    else:
        ctrl_vars = None

    sampler.create_stan_input(
        SPENDS,
        ctrl_vars,
        REVENUE
    )
    obtained_results = sampler.run_sampling(100, 3)

    expected_param_nms = (
        CARRYOVER_TRANSFO_NM_TO_PARAM_NM[carryover_transfo_nm]
        + DIMINUSHING_RETURNS_TRANSFO_NM_TO_PARAM_NM[diminushing_returns_transfo_nm]
        + WITH_CTRL_VARS_TO_PARAM_NM[with_ctrl_vars]
        + ["beta_medias", "tau"]
    )
    expected_param_nms.sort()

    obtained_params_nms = list(obtained_results.keys())
    obtained_params_nms.sort()

    assert expected_param_nms == obtained_params_nms

    for param_nm, values in obtained_results.items():
        if param_nm != "tau":
            assert values.shape == (100,2)
        else:
            assert values.shape == (100,)

    