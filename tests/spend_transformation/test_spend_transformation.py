import numpy as np
import pytest

from bayesian_mmm.spend_transformation.spend_transformation import (
    add_lagged_values_along_z,
    compute_adstock,
    compute_geo_decay,
    compute_hill,
    compute_reach
)

SPENDS = np.array([[10, 20], [0, 8], [1, 30], [5, 40]])

MAX_LAG = 4

LAGGED_SPENDS = np.array([
    [[10,  0, 0, 0], [20,  0, 0, 0]],
    [[ 0, 10, 0, 0], [ 8, 20, 0, 0]],
    [[ 1,  0, 10, 0], [30,  8, 20, 0]],
    [[ 5,  1, 0, 10], [40, 30, 8, 20]]
])

DELAYS = [2.5, 2]
RETAIN_RATES = [0.2, 0.9]
ECS = [0.2, 1]
SLOPES = [1, 4]
HALF_SATURATIONS = [2, 3]


def test_add_lagged_values_along_z():

    print(add_lagged_values_along_z(SPENDS, MAX_LAG), LAGGED_SPENDS)
    
    assert (add_lagged_values_along_z(SPENDS, MAX_LAG) == LAGGED_SPENDS).all()


def test_compute_adstock():

    ADSTOCK_SPENDS = np.array([
        [0.00031371564813652915, 3.796765139897573],
        [0.19607228008533067, 6.726888689563381],
        [4.90183837369808, 13.565290356181823],
        [4.921571087965868, 22.928734700963517]
        ])
    
    assert (compute_adstock(LAGGED_SPENDS, DELAYS, RETAIN_RATES) == ADSTOCK_SPENDS).all()


def test_compute_geo_decay():

    GEO_DECAY_SPENDS = np.array([
        [ 8.012820512820513 ,  5.815644082582146 ],
        [ 1.6025641025641026,  7.560337307356789 ],
        [ 1.1217948717948718, 15.527769700494332 ],
        [ 4.230769230769231 , 25.60628089560919  ]
       ])
    
    assert (compute_geo_decay(LAGGED_SPENDS, RETAIN_RATES) == GEO_DECAY_SPENDS).all()



def test_compute_hill():

    HILL_SPENDS = np.array([
        [0.9803921568627451, 0.9999937500390623],
        [0.                , 0.9997559189650964],
        [0.8333333333333334, 0.9999987654336229],
        [0.9615384615384615, 0.9999996093751526]
        ])
    
    assert (compute_hill(SPENDS, ECS, SLOPES) == HILL_SPENDS).all()



def test_compute_reach():

    REACH_SPENDS = np.array([
        [0.9999999958776927, 1.                ],
        [0.                , 0.9999999999244973],
        [0.7615941559557649, 1.                ],
        [0.9999092042625951, 1.                ]
        ])

    assert (compute_reach(SPENDS, HALF_SATURATIONS) == REACH_SPENDS).all()
