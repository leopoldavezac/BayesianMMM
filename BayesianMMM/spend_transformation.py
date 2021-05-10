import numpy as np


def add_lagged_values_along_z(spends, max_lag):
    
    if type(max_lag) != int:
        raise ValueError("max_lag value must be an int")

    if max_lag < 0:
        raise ValueError("max_lag value must superior or equal 0")
    
    if max_lag >= len(spends):
        raise ValueError("max_lag value superior to nb of obs in spends")

    if not isinstance(spends, np.ndarray):
        raise ValueError("spends must be a numpy matrix")

    if (len(spends.shape) != 2):
        raise ValueError("spends must be a numpy matrix")

    if spends.shape[0] <= spends.shape[1]:
        raise ValueError("nb of media must be less than nb of obs")

    lagged_spends = np.zeros((len(spends), spends.shape[1], max_lag+1))
    
    for lag in range(max_lag+1):

        lagged_spends[:,:,lag] = np.roll(spends, lag, axis=0)
        lagged_spends[:lag,:,lag] = 0

    return lagged_spends


def compute_adstock(spends, peaks, retain_rates):
        
    if not isinstance(spends, np.ndarray):
        raise ValueError("spends must be a numpy tensor")

    if (len(spends.shape) != 3):
        raise ValueError("spends must be a numpy tensor")

    if type(peaks) != list:
        raise ValueError("peaks must be a list of int or float")

    for peak in peaks:

        if type(peak) not in [int, float]:
            raise ValueError("peak value must be an int or a float")

        if peak < 0:
            raise ValueError("peak value must superior or equal 0")

        if peak > spends.shape[2]:
            raise ValueError("peak value must be inferior than max_lag")

    if type(retain_rates) != list:
        raise ValueError("retain_rates must be a list of int or float")

    for retain_rate in retain_rates:

        if type(retain_rate) not in [int, float]:
            raise ValueError("retain_rate value must be an int or a float")

        if (retain_rate < 0) or (retain_rate > 1):
            raise ValueError("retain_rate value must be in [0,1]")


    adstock_spends = np.zeros(spends.shape[:-1])

    for media_index in range(spends.shape[1]):

        lag_weights = np.arange(spends.shape[2])
        lag_weights = np.power(retain_rates[media_index], (lag_weights - peaks[media_index])**2)
        adstock_spends[:,media_index] = np.dot(spends[:,media_index,:], lag_weights) / lag_weights.sum()

    return adstock_spends


def compute_carryover(spends, retain_rates):

    if not isinstance(spends, np.ndarray):
        raise ValueError("spends must be a numpy tensor")

    if (len(spends.shape) != 3):
        raise ValueError("spends must be a numpy tensor")

    if type(retain_rates) != list:
        raise ValueError("retain_rates must be a list of int or float")

    for retain_rate in retain_rates:

        if type(retain_rate) not in [int, float]:
            raise ValueError("retain_rate value must be an int or a float")

        if (retain_rate < 0) or (retain_rate > 1):
            raise ValueError("retain_rate value must be in [0,1]")

    # 
    carryover_spends = np.zeros(spends.shape[:-1])

    for media_index in range(spends.shape[1]):

        lag_weights = np.arange(spends.shape[2])
        lag_weights = np.power(retain_rates[media_index], lag_weights)
        carryover_spends[:,media_index] = np.dot(spends[:,media_index,:], lag_weights) / lag_weights.sum()

    return carryover_spends


def compute_hill(spends, ecs, slopes):

    if not isinstance(spends, np.ndarray):
        raise ValueError("spends must be a numpy matrix")

    if (len(spends.shape) != 2):
        raise ValueError("spends must be a numpy matrix")

    if type(ecs) != list:
        raise ValueError("ecs must be a list of int or float")
    
    for ec in ecs:

        if type(ec) not in [int, float]:
            raise ValueError("ec value must be an int or a float")

        if (ec < 0) or (ec > 1):
            raise ValueError("ec value must be in [0,1]")

    if type(slopes) != list:
        raise ValueError("slopes must be a list of int or float")
    
    for slope in slopes:

        if type(slope) not in [int, float]:
            raise ValueError("slope value must be an int or a float")

        if slope < 0:
            raise ValueError("slope value must be greater than 0")


    hill_spends = np.zeros(spends.shape)

    for media_index in range(spends.shape[1]):
        hill_spends[:, media_index] = 1 / (1 + (spends[:, media_index] / ecs[media_index])**(-slopes[media_index]))

    return hill_spends

def compute_reach(spends, half_saturations):

    if not isinstance(spends, np.ndarray):
        raise ValueError("spends must be a numpy matrix")

    if (len(spends.shape) != 2):
        raise ValueError("spends must be a numpy matrix")

    if type(half_saturations) != list:
        raise ValueError("half_saturations must be a list of int or float")
    
    for half_saturation in half_saturations:

        if type(half_saturation) not in [int, float]:
            raise ValueError("half_saturation value must be an int or a float")

        if half_saturation < 0:
            raise ValueError("half_saturation value must be greater than 0")
    
    reach_spends = np.zeros(spends.shape)

    for media_index in range(spends.shape[1]):
        reach_spends[:, media_index] = (1 - np.exp(-half_saturations[media_index]*spends[:,media_index])) / (1 + np.exp(-half_saturations[media_index]*spends[:,media_index]))

    return reach_spends

