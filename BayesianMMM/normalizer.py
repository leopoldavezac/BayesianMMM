import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler


#add media scaling all together in later version
#add reverse_transform method in later version


class Normalizer:

    def __init__(self, transfo_nm=None, scaler_nm="max_abs"):

        transfo_nm_to_transfo_func = {
            "log":lambda x: np.log(x+1),
            "sqrt":np.sqrt,
            None:lambda x: x
        }

        scaler_nm_to_scaler = {
            "min_max":MinMaxScaler,
            "max_abs":MaxAbsScaler
        }

        if transfo_nm not in transfo_nm_to_transfo_func.keys():
            raise ValueError(str(transfo_nm)+" is not a valid transfo name")

        if scaler_nm not in scaler_nm_to_scaler.keys():
            raise ValueError(str(scaler_nm)+" is not a valid scaler name")

        self.transfo_func = transfo_nm_to_transfo_func[transfo_nm]
        self.scaler = scaler_nm_to_scaler[scaler_nm]


    def fit(self, values):

        transformed_values = self.transfo_func(values)
        self.scaler = self.scaler()
        self.scaler.fit(transformed_values)

    def transform(self, values):
        
        transformed_values = self.transfo_func(values)
        
        return self.scaler.transform(transformed_values)



