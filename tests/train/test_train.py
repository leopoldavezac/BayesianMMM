import pytest
import numpy as np
from mock import patch
from json import load
from pandas import read_csv, set_option

from bayesian_mmm.train import run
from bayesian_mmm.sampling.sampler import Sampler

set_option('mode.chained_assignment', 'raise')

NB_MEDIA = 3
N_ITER = 100 #nb of sampling iteration

OBTAINED_DIR = "./results/"
EXPECTED_DIR = "./tests/train/results/"

@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars",
    (
        ["geo_decay", "reach", False],
        ["geo_decay", "reach", True],
        ["geo_decay", "hill", False],
        ["geo_decay", "hill", True],
        ["adstock", "reach", False],
        ["adstock", "reach", True],
        ["adstock", "hill", False],
        ["adstock", "hill", True],
    )
)
def test_run(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars
):

    sample = create_sampler_run_sampling_output(
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        with_ctrl_vars
    )

    identifier_nm = "%s_%s_with_ctrl_vars_%s" % (
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        str(with_ctrl_vars).lower()
        )

    config_file_nm = "../tests/train/config/"+identifier_nm

    # load config / change experiment nm / save config

    # import json

    # import yaml
    # with open("./tests/train/config/"+identifier_nm+".yaml", "r") as f:
    #     config = yaml.load(f)
    
    # config["EXPERIMENT_NM"] = identifier_nm

    # with open("./tests/train/config/"+identifier_nm+".yaml", "w") as f:
    #     yaml.dump(config, f)

    ###

    with patch.object(Sampler, "run_sampling", new = (lambda x, y, z: sample)):
        run(config_file_nm)

    # load config / change experiment nm / save config
    
    # config["EXPERIMENT_NM"] = "test"

    # with open("./tests/train/config/"+identifier_nm+".yaml", "w") as f:
    #     yaml.dump(config, f)

    # import os
    
    # os.remove("./results/inference_machine/%s.pkl"%identifier_nm)
    # os.remove("./results/normalizer/scaler_predictor_%s.pkl"%identifier_nm)
    # os.remove("./results/normalizer/scaler_target_%s.pkl"%identifier_nm)
    # os.remove("./results/normalizer/args_predictor_%s.json"%identifier_nm)
    # os.remove("./results/normalizer/args_target_%s.json"%identifier_nm)
    # os.remove("./results/contributions_%s.csv"%identifier_nm)
    # os.remove("./results/plot/contribution_analysis_%s.html"%identifier_nm)
    # os.remove("./results/plot/carryover_%s.html"%identifier_nm)
    # os.remove("./results/plot/diminushing_returns_%s.html"%identifier_nm)
    # os.remove("./results/plot/sample_%s.html"%identifier_nm)
    # os.remove("./results/plot/true_vs_pred_%s.html"%identifier_nm)

    # with open("./results/performance_%s.json" % identifier_nm, "r") as f:
    #     perf = json.load(f)

    # os.remove("./results/performance_%s.json" % identifier_nm)

    # with open("./tests/train/results/performance_%s.json" % identifier_nm, "w") as f:
    #     json.dump(perf, f)

    # import pandas as pd

    # pred_vs_true = pd.read_csv("./results/prediction_%s.csv"%identifier_nm)
    # os.remove("./results/prediction_%s.csv" % identifier_nm)

    # pred_vs_true.to_csv("./tests/train/results/prediction_%s.csv" % identifier_nm, index=False)

    # print(pred_vs_true[pred_vs_true.pred.isna()])

    # assert False

    ##

    with open("./results/performance_test.json", "r") as f:
        obtained_performance = load(f)

    with open(
        "./tests/train/results/performance_%s.json" % identifier_nm, 'r'
        ) as f:
        expected_performance = load(f)

    assert obtained_performance == expected_performance
    
    obtained_prediction = read_csv("./results/prediction_test.csv")
    expected_prediction = read_csv(
        "./tests/train/results/prediction_%s.csv" % identifier_nm
        )

    print(
        obtained_prediction[~np.isclose(obtained_prediction.pred.values, expected_prediction.pred.values)],
        expected_prediction[~np.isclose(obtained_prediction.pred.values, expected_prediction.pred.values)]
        )

    assert np.allclose(a=obtained_prediction.true.values, b=expected_prediction.true.values)

    assert np.allclose(a=obtained_prediction.pred.values, b=expected_prediction.pred.values)


def create_sampler_run_sampling_output(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars
):

    if with_ctrl_vars:
        nb_ctrl = 1
    else:
        nb_ctrl = 0

    np.random.seed(2021)

    sample = {
        "tau":np.random.rand(N_ITER),
        "beta_medias":np.random.rand(N_ITER, NB_MEDIA)
    }

    if carryover_transfo_nm == "adstock":
        sample["retain_rate"] = np.random.rand(N_ITER, NB_MEDIA)
        sample["delay"] = np.random.rand(N_ITER, NB_MEDIA)
    elif carryover_transfo_nm == "geo_decay":
        sample["retain_rate"] = np.random.rand(N_ITER, NB_MEDIA)

    if diminushing_returns_transfo_nm == "hill":
        sample["ec"] = np.random.rand(N_ITER, NB_MEDIA)
        sample["slope"] = np.random.rand(N_ITER, NB_MEDIA)
    elif diminushing_returns_transfo_nm == "reach":
        sample["half_saturation"] = np.random.rand(N_ITER, NB_MEDIA)

    if nb_ctrl > 0:
        sample["gamma_ctrl"] = np.random.rand(N_ITER, nb_ctrl)

    return sample