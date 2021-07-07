from typing import Dict, List, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy import ndarray

class SampleVisualizor:

    def __init__(self, media_nms: List[str], ctrl_nms: Union[List[str], None]) -> None:    
        
        self.__media_nms = media_nms
        self.__ctrl_nms = ctrl_nms if ctrl_nms else []

        self.__row_index = 1
        self.__col_index = 1

    def write_fig(self, sample: Dict, name:str) -> None:
        
        self.__create_fig(sample)
        self.__fig.write_html("results/plot/sample_"+name+".html", auto_open=False)

    def __create_fig(self, sample: Dict) -> None:

        param_nms = list(sample.keys())

        self.__compute_plot_ncols(param_nms)
        self.__compute_plot_nrows()
        
        spend_related_param_nms = self.__get_spend_related_param_nms(param_nms)
        titles = self.__get_subplot_titles(spend_related_param_nms)
        self.__fig = self.__get_empty_fig(titles)

        self.__add_subplot_spend_related_params(sample, spend_related_param_nms)
        if self.__ctrl_nms:
            self.__add_subplot_gamma_ctrl(sample["gamma_ctrl"])
        self.__add_subplot_tau(sample["tau"])

        self.__fig.update_layout(showlegend=False)

    def __compute_plot_ncols(self, param_nms: List[str]) -> None:
        
        self.__ncols = len(
            [param_nm for param_nm in param_nms if param_nm not in ["gamma_ctrl", "tau"]]
            )
        
    def __compute_plot_nrows(self) -> None:

            self.__nrows = len(self.__media_nms) + (len(self.__ctrl_nms) + 1) // self.__ncols + 1

    def __get_spend_related_param_nms(self, param_nms: List[str]) -> List[str]:

            return [param_nm for param_nm in param_nms if param_nm not in ["gamma_ctrl", "tau"]]

    def __get_subplot_titles(self, spend_related_param_nms: List[str]) -> List[str]:
        
        titles = []

        for media_nm in self.__media_nms:
            for spend_related_param_nm in spend_related_param_nms:
                titles.append(media_nm + " " + spend_related_param_nm)
        
        for ctrl_nm in self.__ctrl_nms:
            titles.append(ctrl_nm + " gamma_ctrl")

        titles.append("tau")

        return titles

    def __get_empty_fig(self, titles: List[str]) -> go.Figure:

        return make_subplots(rows=self.__nrows, cols=self.__ncols, subplot_titles=titles)
    
    def __add_subplot_spend_related_params(
        self, sample : Dict,
        spend_related_params: List[str]
        ) -> None:

        for media_index, _ in enumerate(self.__media_nms):
            for spend_related_param in spend_related_params:
                self.__fig.add_trace(
                    go.Histogram(x=sample[spend_related_param][:,media_index]),
                    row=self.__row_index,
                    col=self.__col_index
                )
                self.__update_subplot_index()
    
    def __add_subplot_gamma_ctrl(self, sample_gamma_ctrl: ndarray) -> None:

        for ctrl_index, _ in enumerate(self.__ctrl_nms):
            self.__fig.add_trace(
                go.Histogram(x=sample_gamma_ctrl[:,ctrl_index]),
                row=self.__row_index,
                col=self.__col_index
            )
            self.__update_subplot_index()

    def __add_subplot_tau(self, sample_tau: ndarray) -> None:
        
        self.__fig.add_trace(
            go.Histogram(x=sample_tau),
            row=self.__row_index,
            col=self.__col_index
        )

    
    def __update_subplot_index(self) -> None:

        self.__col_index += 1

        if self.__col_index > self.__ncols:
            self.__row_index += 1
            self.__col_index = 1


