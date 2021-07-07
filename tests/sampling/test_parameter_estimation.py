import pytest
import numpy as np

from bayesian_mmm.sampling.parameter_estimation import estimate_parameters

@pytest.mark.parametrize(
    "carryover_transfo_nm,diminushing_returns_transfo_nm,media_nms,ctrl_nms,estimator_nm",
    [
        ("adstock","hill",["radio", "tv"],[],"mean"),
        ("adstock","hill",["radio","tv","facebook"],["ggtrnds"],"median"),
        ("adstock","reach",["radio","tv","facebook"],["ggtrnds","consumer_index"],"mean"),
        ("adstock","reach",["radio", "tv"],[],"median"),
        ("geo_decay","hill",["radio", "tv"],["ggtrnds"],"mean"),
        ("geo_decay","hill",["radio","tv","facebook"],["ggtrnds","consumer_index"],"median"),
        ("geo_decay","reach",["radio", "tv"],["ggtrnds"],"mean"),
        ("geo_decay","reach",["radio", "tv"],[],"median")
    ]
)
def test_estimate_parameters(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    media_nms,
    ctrl_nms,
    estimator_nm
):

    nb_media = len(media_nms)
    nb_ctrl = len(ctrl_nms)

    mean = 2
    std = 0.5

    sample = {
        "tau":np.random.normal(mean, std, (100,)),
        "beta_medias":np.random.normal(mean, std, (100, nb_media))
    }

    if carryover_transfo_nm == "adstock":
        sample["retain_rate"] = np.random.normal(mean, std, (100, nb_media))
        sample["delay"] = np.random.normal(mean, std, (100, nb_media))
    elif carryover_transfo_nm == "geo_decay":
        sample["retain_rate"] = np.random.normal(mean, std, (100, nb_media))

    if diminushing_returns_transfo_nm == "hill":
        sample["ec"] = np.random.normal(mean, std, (100, nb_media))
        sample["slope"] = np.random.normal(mean, std, (100, nb_media))
    elif diminushing_returns_transfo_nm == "reach":
        sample["half_saturation"] = np.random.normal(mean, std, (100, nb_media))

    if nb_ctrl > 0:
        sample["gamma_ctrl"] = np.random.normal(mean, std, (100, nb_ctrl))

    obtained_results = estimate_parameters(sample, estimator_nm)

    for value in obtained_results.values():
        if type(value) == np.ndarray:
            for v in value:
                assert mean == round(v) #normal distribution -> mean == median
        else:
            assert mean == round(value) #normal distribution -> mean == median
    