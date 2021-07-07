from typing import Union
import numpy as np
from numpy import ndarray
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from pickle import dump as pkl_dump, load as pkl_load
from json import dump as json_dump, load as json_load

from bayesian_mmm.utilities.utilities import check_ndarray_is_matrix


class Normalizer:

    def __init__(self, transfo_nm: str = None, scaler_nm: str = "max_abs") -> None:

        transfo_nm_to_transfo_func = {
            "log": lambda x: np.log(x+1),
            "sqrt": np.sqrt,
            None: lambda x: x
        }

        transfo_nm_to_reverse_transfo_func = {
            "log": lambda x: np.exp(x) -1,
            "sqrt": lambda x: np.power(x,2),
            None: lambda x: x
        }

        scaler_nm_to_scaler = {
            "min_max":MinMaxScaler,
            "max_abs":MaxAbsScaler
        }

        if transfo_nm not in transfo_nm_to_transfo_func.keys():
            raise ValueError(str(transfo_nm)+" is not a valid transfo name")

        if scaler_nm not in scaler_nm_to_scaler.keys():
            raise ValueError(str(scaler_nm)+" is not a valid scaler name")

        self.__transfo_nm = transfo_nm
        self.__scaler_nm = scaler_nm

        self.__transfo_func = transfo_nm_to_transfo_func[transfo_nm]
        self.__reverse_transfo_func = transfo_nm_to_reverse_transfo_func[transfo_nm]
        self.__scaler = scaler_nm_to_scaler[scaler_nm]


    def __check_non_negative_values_with_log_transfo(self, values: ndarray) -> bool:

        if self.__transfo_nm == "log":
            if len(values[values < 0]) > 0:
                raise ValueError("Log transfo with negative values")


    def fit(self, values: ndarray) -> None:

        check_ndarray_is_matrix(values, "values")
        self.__check_non_negative_values_with_log_transfo(values)

        transformed_values = self.__transfo_func(values)
        self.__scaler = self.__scaler()
        self.__scaler.fit(transformed_values)

    def transform(self, values: ndarray) -> ndarray:

        check_ndarray_is_matrix(values, "values")
        self.__check_non_negative_values_with_log_transfo(values)
        
        transformed_values = self.__transfo_func(values)
        
        return self.__scaler.transform(transformed_values)

    def reverse_transform(self, normalized_values: ndarray) -> ndarray:

        check_ndarray_is_matrix(normalized_values, "normalized_values")
        
        descaled_values = self.__scaler.inverse_transform(normalized_values)

        return self.__reverse_transfo_func(descaled_values)

    def save(self, name: str):

        with open("./results/normalizer/scaler_%s.pkl" % name, "wb") as f:
            pkl_dump(self.__scaler, f)

        with open("./results/normalizer/args_%s.json" % name, "w") as f:
            json_dump({
                "transfo_nm": self.__transfo_nm,
                "scaler_nm": self.__scaler_nm
            }, f)

    def set_fitted_scaler(self, scaler: Union[MaxAbsScaler, MinMaxScaler]) -> None:

        self.__scaler = scaler



def load_normalizer(name: str) -> Normalizer:

    with open("./results/normalizer/scaler_%s.pkl" % name, "rb") as f:
            scaler = pkl_load(f)

    with open("./results/normalizer/args_%s.json" % name, "r") as f:
        init_args = json_load(f)

    normalizer = Normalizer(**init_args)
    normalizer.set_fitted_scaler(scaler)



