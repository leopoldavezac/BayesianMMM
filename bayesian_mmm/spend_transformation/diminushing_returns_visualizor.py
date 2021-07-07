from typing import Dict, List
from numpy import ndarray, zeros
import plotly.graph_objects as go

from bayesian_mmm.spend_transformation.spend_transformation import (
    compute_hill,
    compute_reach
)

class DiminushingReturnsVisualizor:

    def __init__(self, param_nm_to_val: Dict, media_nms: List[str]) -> None:

        if "ec" in param_nm_to_val.keys():
            self.__transfo_func = compute_hill
            self.__transfo_params = {
                "ecs":param_nm_to_val["ec"],
                "slopes":param_nm_to_val["slope"]
                }
        else:
            self.__transfo_func = compute_reach
            self.__transfo_params = {"half_saturations":param_nm_to_val["half_saturation"]}

        self.__media_nms = media_nms
        

    def write_fig(self, spends: ndarray, name: str) -> None:

        transformed_spends = self.__transfo_func(spends, **self.__transfo_params)
        
        fig = go.Figure()

        for media_index, media_nm in enumerate(self.__media_nms):

            trace_data = zeros((spends.shape[0], 2))
            trace_data[:,0] = spends[:,media_index]
            trace_data[:,1] = transformed_spends[:,media_index]
            trace_data = trace_data[trace_data[:,0].argsort()]

            fig.add_trace(
                go.Scatter(
                    x = trace_data[:,0],
                    y = trace_data[:,1],
                    name = media_nm
                )
            )

        fig.write_html("results/plot/diminushing_returns_%s.html" % name, auto_open=False)