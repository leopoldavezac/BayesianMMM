
from bayesian_mmm.utilities.utilities import load_config
from typing import Union
from numpy import ndarray
from bayesian_mmm.normalizer.normalizer import load_normalizer
from bayesian_mmm.inference_machine.inference_machine import load_inference_machine



def predict(spends: ndarray, ctrl_vars: Union[ndarray, None]):

    config = load_config("predict")

    target_normalizer = load_normalizer("target_"+config["EXPERIMENT_NM"])
    inference_machine = load_inference_machine(config["EXPERIMENT_NM"])

    pred = inference_machine.predict(spends, ctrl_vars)
    pred = target_normalizer.denormalize(pred)

    return pred


