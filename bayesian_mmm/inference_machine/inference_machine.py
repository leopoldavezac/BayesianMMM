from pickle import dump, load
from typing import Dict, Union
from numpy import ndarray, dot

from bayesian_mmm.spend_transformation.spend_transformation import (
    add_lagged_values_along_z,
    compute_adstock,
    compute_geo_decay,
    compute_hill,
    compute_reach
    )
from bayesian_mmm.utilities.utilities import check_ndarray_is_matrix


class InferenceMachine:

    def __init__(self, param_nm_to_val: Dict, max_lag: int) -> None:

        if "delay" in param_nm_to_val.keys():
            self.__carryover_func = compute_adstock
            self.__carryover_params = {
                "delays":param_nm_to_val["delay"], "retain_rates":param_nm_to_val["retain_rate"]
                }
        else:
            self.__carryover_func = compute_geo_decay
            self.__carryover_params = {
                "retain_rates":param_nm_to_val["retain_rate"]
                }

        if "ec" in param_nm_to_val.keys():
            self.__diminushing_returns_func = compute_hill
            self.__diminushing_returns_params = {
                "ecs":param_nm_to_val["ec"], "slopes":param_nm_to_val["slope"]
                }
        else:
            self.__diminushing_returns_func = compute_reach
            self.__diminushing_returns_params = {
                "half_saturations":param_nm_to_val["half_saturation"]
                }

        try:
            self._gamma_ctrl = param_nm_to_val["gamma_ctrl"]
        except(KeyError):
            pass

        self._tau = param_nm_to_val["tau"]
        self._beta_medias = param_nm_to_val["beta_medias"]

        self.__max_lag = max_lag


    def predict(self, spends: ndarray, ctrl_vars: Union[ndarray, None]) -> ndarray:

        transformed_spends = self._get_transformed_spends(spends)        

        y = dot(transformed_spends, self._beta_medias) + self._tau

        if type(ctrl_vars) == ndarray:
            check_ndarray_is_matrix(ctrl_vars, "ctrl_vars")
            y += dot(ctrl_vars, self._gamma_ctrl)

        return y

    def _get_transformed_spends(self, spends: ndarray) -> ndarray:

        lagged_spends = add_lagged_values_along_z(spends, self.__max_lag)

        transformed_spends = self.__carryover_func(lagged_spends, **self.__carryover_params)
        transformed_spends = self.__diminushing_returns_func(
            transformed_spends, **self.__diminushing_returns_params
            )

        return transformed_spends

   

def save_inference_machine(inference_machine: InferenceMachine, name: str) -> None:

    with open("./results/inference_machine/%s.pkl" % name, "wb") as f:
        dump(inference_machine, f)


def load_inference_machine(name: str) -> InferenceMachine:

    with open("./results/inference_machine/%s.pkl" % name, "rb") as f:
        inference_machine = load(f)
    
    return inference_machine