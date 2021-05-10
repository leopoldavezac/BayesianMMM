import numpy as np
import pytest


from spend_transformation import (
    add_lagged_values_along_z,
    compute_adstock,
    compute_carryover,
    compute_hill,
    compute_reach
)

def test_add_lagged_values_along_z():

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    MAX_LAG = 3
    LAGGED_SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])
    
    assert (add_lagged_values_along_z(spends=SPENDS, max_lag=MAX_LAG) == LAGGED_SPENDS).all()


@pytest.mark.parametrize(
    "spends",
    [
        "np.array([])",
        np.array([[10, 20, 0, 8], [1, 30, 5, 40]])
        ]
    )
def test_add_lagged_values_along_z_input_error_spends(spends):
    
    MAX_LAG = 2

    with pytest.raises(ValueError):
        add_lagged_values_along_z(spends=spends, max_lag=MAX_LAG)


@pytest.mark.parametrize(
    "max_lag",
    [-1, 4, "12"]
    )
def test_add_lagged_values_along_z_input_error_max_lag(max_lag):

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    
    with pytest.raises(ValueError):
        add_lagged_values_along_z(spends=SPENDS, max_lag=max_lag)



def test_compute_adstock():

    PEAKS = [2.5, 2]
    RETAIN_RATES = [0.2, 0.9]
    SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])

    ADSTOCK_SPENDS = np.array([
        [0.00031371564813652915, 3.796765139897573],
        [0.19607228008533067, 6.726888689563381],
        [4.90183837369808, 13.565290356181823],
        [4.921571087965868, 22.928734700963517]
        ])
    
    assert (compute_adstock(spends=SPENDS, peaks=PEAKS, retain_rates=RETAIN_RATES) == ADSTOCK_SPENDS).all()


@pytest.mark.parametrize(
    "spends",
    [
        "np.array([])",
        np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
        ]
    )
def test_compute_adstock_input_error_spends(spends):
    
    PEAKS = [2, 1]
    RETAIN_RATES = [0.1, 0.2]

    with pytest.raises(ValueError):
        compute_adstock(spends=spends, peaks=PEAKS, retain_rates=RETAIN_RATES)


@pytest.mark.parametrize(
    "peaks",
    [[-1, 0], [0, 5], "[0, 1]"]
    )
def test_compute_adstock_input_error_peaks(peaks):

    SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])

    RETAIN_RATES = [0.2, 1]
    
    with pytest.raises(ValueError):
        compute_adstock(spends=SPENDS, peaks=peaks, retain_rates=RETAIN_RATES)



@pytest.mark.parametrize(
    "retain_rates",
    [[-1, 0], [2, 0.2], "[0, 1]"]
    )
def test_compute_adstock_input_error_retain_rates(retain_rates):

    SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])

    PEAKS = [0.2, 1]
    
    with pytest.raises(ValueError):
        compute_adstock(spends=SPENDS, peaks=PEAKS, retain_rates=retain_rates)


def test_compute_carryover():

    RETAIN_RATES = [0.2, 0.9]
    SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])

    CARRYOVER_SPENDS = np.array([
        [ 8.012820512820513 ,  5.815644082582146 ],
        [ 1.6025641025641026,  7.560337307356789 ],
        [ 1.1217948717948718, 15.527769700494332 ],
        [ 4.230769230769231 , 25.60628089560919  ]
       ])
    
    assert (compute_carryover(spends=SPENDS, retain_rates=RETAIN_RATES) == CARRYOVER_SPENDS).all()



@pytest.mark.parametrize(
    "retain_rates",
    [[-1, 0], [2, 0.2], "[0, 1]"]
    )
def test_compute_carryover_input_error_retain_rates(retain_rates):

    SPENDS = np.array([
            [[10,  0, 0, 0], [20,  0, 0, 0]],
            [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
            [[ 1,  0, 10, 0], [30,  8, 20, 0]],
            [[ 5,  1, 0, 10], [40, 30, 8, 20]]
        ])
    
    with pytest.raises(ValueError):
        compute_carryover(spends=SPENDS, retain_rates=retain_rates)



def test_compute_hill():

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    ECS = [0.2, 1]
    SLOPES = [1, 4]
    HILL_SPENDS = np.array([
        [0.9803921568627451, 0.9999937500390623],
        [0.                , 0.9997559189650964],
        [0.8333333333333334, 0.9999987654336229],
        [0.9615384615384615, 0.9999996093751526]
        ])
    
    assert (compute_hill(spends=SPENDS, ecs=ECS, slopes=SLOPES) == HILL_SPENDS).all()



@pytest.mark.parametrize(
    "spends",
    [
        "this is an array",
        np.arange(9),
    ]
    )
def test_compute_hill_input_error_spends(spends):

    ECS = [0.2, 1]
    SLOPES = [1, 4]    

    with pytest.raises(ValueError):
        compute_hill(spends=spends, ecs=ECS, slopes=SLOPES)



@pytest.mark.parametrize(
    "ecs",
    [
        "0.2",
        -1,
        2
    ]
    )
def test_compute_hill_input_error_ecs(ecs):

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    SLOPES = [1, 4]    

    with pytest.raises(ValueError):
        compute_hill(spends=SPENDS, ecs=ecs, slopes=SLOPES)



@pytest.mark.parametrize(
    "slopes",
    [
        "0.2",
        -1
    ]
    )
def test_compute_hill_input_error_slopes(slopes):

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    ECS = [0.2, 1]

    with pytest.raises(ValueError):
        compute_hill(spends=SPENDS, ecs=ECS, slopes=slopes)






def test_compute_reach():

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])
    HALF_SATURATIONS = [2, 3]
    REACH_SPENDS = np.array([
        [0.9999999958776927, 1.                ],
        [0.                , 0.9999999999244973],
        [0.7615941559557649, 1.                ],
        [0.9999092042625951, 1.                ]
        ])

    assert (compute_reach(spends=SPENDS, half_saturations=HALF_SATURATIONS) == REACH_SPENDS).all()


@pytest.mark.parametrize(
    "spends",
    [
        "this is an array",
        np.arange(9),
    ]
    )
def test_compute_reach_input_error_spends(spends):

    HALF_SATURATIONS = [2, 3]    

    with pytest.raises(ValueError):
        compute_reach(spends=spends, half_saturations=HALF_SATURATIONS)



@pytest.mark.parametrize(
    "half_saturations",
    [
        "0.2",
        -1
    ]
    )
def test_compute_reach_input_error_half_saturations(half_saturations):

    SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])    

    with pytest.raises(ValueError):
        compute_reach(spends=SPENDS, half_saturations=half_saturations)


