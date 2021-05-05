import pytest
import numpy as np

from normalizer import Normalizer


def test_normalizer_init_error_handling_input_transfo_nm():

    UNKNOWN_TRANSFO_NM = "aezrergh"

    with pytest.raises(Exception):
        Normalizer(transfo_nm=UNKNOWN_TRANSFO_NM)



def test_normalizer_init_error_handling_input_scaler_nm():

    UNKNOWN_TRANSFO_NM = "aezrergh"

    with pytest.raises(Exception):
        Normalizer(scaler_nm=UNKNOWN_TRANSFO_NM)





# def test_normalizer_init_transfo():

#     normalizer = Normalizer()
#     assert normalizer.transfo_func in [np.log, np.exp, np.sqrt, np.power]

    