from typing import List
import numpy as np

from bayesian_mmm.utilities.utilities import check_ndarray_is_matrix, check_ndarray_is_tensor

def add_lagged_values_along_z(spends: np.ndarray, max_lag: int) -> np.ndarray:
    
    if max_lag < 0:
        raise ValueError("max_lag value must superior or equal 0")
    
    if max_lag > len(spends):
        raise ValueError("max_lag value superior to nb of obs in spends")

    check_ndarray_is_matrix(spends, "spends")

    lagged_spends = np.zeros((len(spends), spends.shape[1], max_lag))
    
    for lag in range(max_lag):

        lagged_spends[:,:,lag] = np.roll(spends, lag, axis=0)
        lagged_spends[:lag,:,lag] = 0

    return lagged_spends


def compute_adstock(
    lagged_spends: np.ndarray, delays: List[float], retain_rates: List[float]
    ) -> np.ndarray:
        
    check_ndarray_is_tensor(lagged_spends, "lagged_spends")

    for delay in delays:

        if delay < 0:
            raise ValueError("delay value must superior or equal 0")

        if delay > lagged_spends.shape[2]:
            raise ValueError("delay value must be inferior than max_lag")

    for retain_rate in retain_rates:

        if (retain_rate < 0) or (retain_rate > 1):
            raise ValueError("retain_rate value must be in [0,1]")


    adstock_spends = np.zeros(lagged_spends.shape[:-1])

    for media_index in range(lagged_spends.shape[1]):

        lag_weights = np.arange(lagged_spends.shape[2])
        lag_weights = np.power(retain_rates[media_index], (lag_weights - delays[media_index])**2)
        adstock_spends[:,media_index] = np.dot(lagged_spends[:,media_index,:], lag_weights) / lag_weights.sum()

    return adstock_spends


def compute_geo_decay(lagged_spends: np.ndarray, retain_rates: List[float]) -> np.ndarray:

    check_ndarray_is_tensor(lagged_spends, "lagged_spends")

    for retain_rate in retain_rates:

        if (retain_rate < 0) or (retain_rate > 1):
            raise ValueError("retain_rate value must be in [0,1]")

    # 
    carryover_spends = np.zeros(lagged_spends.shape[:-1])

    for media_index in range(lagged_spends.shape[1]):

        lag_weights = np.arange(lagged_spends.shape[2])
        lag_weights = np.power(retain_rates[media_index], lag_weights)
        carryover_spends[:,media_index] = np.dot(lagged_spends[:,media_index,:], lag_weights) / lag_weights.sum()

    return carryover_spends


def compute_hill(spends: np.ndarray, ecs: List[float], slopes: List[float]) -> np.ndarray:

    check_ndarray_is_matrix(spends, "spends")

    for ec in ecs:

        if (ec < 0) or (ec > 1):
            raise ValueError("ec value must be in [0,1]")

    for slope in slopes:

        if slope < 0:
            raise ValueError("slope value must be greater than 0")


    hill_spends = np.zeros(spends.shape)

    for media_index in range(spends.shape[1]):
        hill_spends[:, media_index] = 1 / (1 + (spends[:, media_index] / ecs[media_index])**(-slopes[media_index]))

    return hill_spends

def compute_reach(spends: np.ndarray, half_saturations: List[float]) -> np.ndarray:

    check_ndarray_is_matrix(spends, "spends")

    for half_saturation in half_saturations:

        if half_saturation < 0:
            raise ValueError("half_saturation value must be greater than 0")
    
    reach_spends = np.zeros(spends.shape)

    for media_index in range(spends.shape[1]):
        reach_spends[:, media_index] = (1 - np.exp(-half_saturations[media_index]*spends[:,media_index])) / (1 + np.exp(-half_saturations[media_index]*spends[:,media_index]))

    return reach_spends

