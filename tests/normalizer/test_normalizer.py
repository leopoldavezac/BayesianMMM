import pytest
import numpy as np

from bayesian_mmm.normalizer.normalizer import Normalizer

TEST_VALUES = np.array(
    [
        [0, 0],
        [5, 15],
        [10, 60]
    ]
)


def test_normalizer_init_error_handling_input_transfo_nm():

    UNKNOWN_TRANSFO_NM = "aezrergh"

    with pytest.raises(ValueError):
        Normalizer(transfo_nm=UNKNOWN_TRANSFO_NM)



def test_normalizer_init_error_handling_input_scaler_nm():

    UNKNOWN_SCALER_NM = "aezrergh"

    with pytest.raises(ValueError):
        Normalizer(scaler_nm=UNKNOWN_SCALER_NM)



@pytest.mark.parametrize(
    "scaler_nm,transfo_nm,normalized_values",
    [
        ("max_abs", None, np.array([[0, 0],[0.5, 0.25], [1, 1]])),
        ("min_max", "log", np.array([[0, 0],[0.7472217363092141, 0.6744523947580044], [1, 1]])),
        ("max_abs", "sqrt", np.array([[0, 0],[0.7071067811865475, 0.5], [1, 1]]))
        ]
    )
def test_normalizer_transform(scaler_nm, transfo_nm, normalized_values):

    normalizer = Normalizer(transfo_nm=transfo_nm, scaler_nm=scaler_nm)
    normalizer.fit(TEST_VALUES)
    
    assert (normalizer.transform(TEST_VALUES) == normalized_values).all()


@pytest.mark.parametrize(
    "scaler_nm,transfo_nm",
    [
        ("max_abs", None),
        ("min_max", "log"),
        ("max_abs", "sqrt")
        ]
    )
def test_reverse_transform(scaler_nm, transfo_nm):

    normalizer = Normalizer(transfo_nm=transfo_nm, scaler_nm=scaler_nm)
    normalizer.fit(TEST_VALUES)
    normalized_values = normalizer.transform(TEST_VALUES)

    assert np.allclose(TEST_VALUES, normalizer.reverse_transform(normalized_values)) #all close to ignore precision differences

