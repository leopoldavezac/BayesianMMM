from typing import Dict, List
from numpy import ndarray, arange
import plotly.graph_objects as go

from bayesian_mmm.spend_transformation.spend_transformation import (
    add_lagged_values_along_z,
    compute_adstock,
    compute_geo_decay
)
from bayesian_mmm.utilities.utilities import generate_random_color

class CarryoverVisualizor:

    def __init__(self, param_nm_to_val: Dict, media_nms: List[str], max_lag: int) -> None:

        if "delay" in param_nm_to_val.keys():
            self.__transfo_func = compute_adstock
            self.__transfo_params = {
                "delays":param_nm_to_val["delay"],
                "retain_rates":param_nm_to_val["retain_rate"]
                }
        else:
            self.__transfo_func = compute_geo_decay
            self.__transfo_params = {"retain_rates":param_nm_to_val["retain_rate"]}

        self.__media_nms = media_nms
        self.__max_lag = max_lag
        

    def write_fig(self, spends: ndarray, name: str) -> None:

        lagged_spends = add_lagged_values_along_z(spends, self.__max_lag)
        transformed_spends = self.__transfo_func(lagged_spends, **self.__transfo_params)

        xvalues = arange(spends.shape[0])
        
        fig = go.Figure()

        for media_index, media_nm in enumerate(self.__media_nms):

            media_color = generate_random_color()

            fig.add_trace(
                go.Scatter(
                    x = xvalues,
                    y = spends[:,media_index],
                    name = media_nm,
                    line = dict(color=media_color)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x = xvalues,
                    y = transformed_spends[:,media_index],
                    name = "carryover "+ media_nm,
                    line = dict(color=media_color)
                )
            )

        fig.write_html("results/plot/carryover_%s.html" % name, auto_open=False)