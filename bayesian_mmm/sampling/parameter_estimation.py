from typing import Dict
import numpy as np

def estimate_parameters(sample: Dict, estimator_nm: str) -> Dict:

    if estimator_nm not in ["mean", "median"]:
        raise(ValueError("estimator_nm must be 'mean' or 'median'"))

    param_estimation = {}

    if estimator_nm == "median":
        estimatation_func = np.median
    elif estimator_nm == "mean":
        estimatation_func = np.mean

    for param_nm, values in sample.items():
        param_estimation[param_nm] = estimatation_func(values, axis=0)

    return param_estimation
