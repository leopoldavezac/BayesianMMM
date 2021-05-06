import pytest
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler


from normalizer import Normalizer




def test_normalizer_init_error_handling_input_transfo_nm():

    UNKNOWN_TRANSFO_NM = "aezrergh"

    with pytest.raises(ValueError):
        Normalizer(transfo_nm=UNKNOWN_TRANSFO_NM)



def test_normalizer_init_error_handling_input_scaler_nm():

    UNKNOWN_SCALER_NM = "aezrergh"

    with pytest.raises(ValueError):
        Normalizer(scaler_nm=UNKNOWN_SCALER_NM)



@pytest.mark.parametrize(
    "transfo_nm,transfo_func",
    [
        ("log",lambda x: np.log(x+1)),
        ("sqrt", np.sqrt),
        (None, lambda x: x),
        ]
    )
def test_normalizer_init_transfo(transfo_nm, transfo_func):

    normalizer = Normalizer(transfo_nm=transfo_nm)

    if transfo_nm == "sqrt":
        assert normalizer.transfo_func == transfo_func
    else:
        assert normalizer.transfo_func.__code__.co_code == transfo_func.__code__.co_code


@pytest.mark.parametrize(
    "scaler_nm,scaler",
    [
        ("max_abs", MaxAbsScaler),
        ("min_max", MinMaxScaler),
        ]
    )
def test_normalizer_init_scaler(scaler_nm, scaler):

    normalizer = Normalizer(scaler_nm=scaler_nm)
    assert normalizer.scaler == scaler


@pytest.mark.parametrize(
    "scaler_nm,transfo_nm,normalized_values",
    [
        ("max_abs", None, np.array([[0, 0],[0.5, 0.25], [1, 1]])),
        ("min_max", "log", np.array([[0, 0],[0.7472217363092141, 0.6744523947580044], [1, 1]])),
        ("max_abs", "sqrt", np.array([[0, 0],[0.7071067811865475, 0.5], [1, 1]]))
        ]
    )
def test_normalizer_transform(scaler_nm, transfo_nm, normalized_values):

    TEST_VALUES = np.array(
        [
            [0, 0],
            [5, 15],
            [10, 60]
        ]
    )

    normalizer = Normalizer(transfo_nm=transfo_nm, scaler_nm=scaler_nm)
    normalizer.fit(TEST_VALUES)
    
    assert (normalizer.transform(TEST_VALUES) == normalized_values).all()
