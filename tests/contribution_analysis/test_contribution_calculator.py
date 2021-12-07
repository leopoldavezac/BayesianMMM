import pytest
import numpy as np

from bayesian_mmm.contribution_analysis.contribution_calculator import ContributionCalculator
from bayesian_mmm.normalizer.normalizer import Normalizer

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

MEDIA_NMS = ["radio", "tv"]
CTRL_NMS = ["consumer_index", "ggtrnds"]
REVENUE_NORMALIZER = Normalizer # non instanciated neither fitted -> not required to test get normalized contributions

@pytest.mark.parametrize(
    "carryover_transfo_nm,diminushing_returns_transfo_nm,with_ctrl_vars,max_lag",
    [
        ("adstock","hill",True, 2),
        ("adstock","hill",False, 3),
        ("adstock","reach",True, 2),
        ("adstock","reach",False, 3),
        ("geo_decay","hill",True, 2),
        ("geo_decay","hill",False, 3),
        ("geo_decay","reach",True, 2),
        ("geo_decay","reach",False, 3)
    ]
)
def test_get_normalized_contributions(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars,
    max_lag
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
        ctrl_vars = CTRL_VARS
        ctrl_nms = CTRL_NMS
    else:
        ctrl_vars = None
        ctrl_nms = []

    contribution_calculator = ContributionCalculator(
        param_nm_to_val,
        max_lag,
        REVENUE_NORMALIZER,
        MEDIA_NMS,
        ctrl_nms
        )
    normalized_contributions = contribution_calculator._ContributionCalculator__get_normalized_results(
        SPENDS, ctrl_vars
        )
    normalized_pred = contribution_calculator.predict(
        SPENDS, ctrl_vars
    )

    assert np.allclose(normalized_contributions.sum(axis=1), normalized_pred) #allclose to ignore precision differences