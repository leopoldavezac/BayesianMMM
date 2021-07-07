from typing import Dict, Union
from numpy import array, ndarray

from bayesian_mmm.sampling.stan_model_wrapper import StanModelWrapper
from bayesian_mmm.utilities.utilities import (
    check_ndarray_is_matrix, check_ndarray_is_vector
)
from bayesian_mmm.spend_transformation.spend_transformation import add_lagged_values_along_z

class Sampler:

    def __init__(self, stan_model: StanModelWrapper, max_lag: int) -> None:
        
        self.__stan_model = stan_model
        self.__max_lag = max_lag

    def create_stan_input(
        self,
        spends: array,
        ctrl_vars: Union[array, None],
        revenue: array,
        ):

        check_ndarray_is_matrix(spends, "lagged_spends")
        check_ndarray_is_vector(revenue, "revenue")

        lagged_spends = add_lagged_values_along_z(spends, self.__max_lag)

        self.__args = {
            "N":lagged_spends.shape[0],
            "Y":revenue,
            "max_lag":lagged_spends.shape[2],
            "num_media":lagged_spends.shape[1],
            "X_media":lagged_spends
        }

        if type(ctrl_vars) == ndarray:

            check_ndarray_is_matrix(ctrl_vars, "ctrl_vars")
            self.__args.update({
                "num_ctrl":ctrl_vars.shape[1],
                "X_ctrl":ctrl_vars
            })

    def run_sampling(self, n_iter: int, chains: int) -> Dict:

        sampling_results = self.__stan_model.sample(
            self.__args,
            n_iter,
            chains
        )

        RELEVANT_PARAM_NMS_UNIVERSE = [
            "retain_rate", "delay",
            "ec", "slope", "half_saturation",
            "beta_medias", "gamma_ctrl", "tau"
        ]

        sampling_relevant_results = {}

        for param_nm, sampled_values in sampling_results.items():
            if param_nm in RELEVANT_PARAM_NMS_UNIVERSE:
                sampling_relevant_results[param_nm] = sampled_values[-n_iter:]
        
        return sampling_relevant_results


        
