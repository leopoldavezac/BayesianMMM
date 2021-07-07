import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas import DataFrame, Grouper, Series

from bayesian_mmm.utilities.utilities import generate_random_color

class ContributionVisualizor:


    def __init__(self, contributions: DataFrame, spends: DataFrame) -> None:

        # index of contributions and spends must be datetime
        self.__contributions = self.__group_at_month_level(contributions)
        self.__spends = self.__group_at_month_level(spends)

        self.__incremental_to_spends_ratio = self.__get_incremental_to_spends_ratio()

        self.__set_color_by_var()

    
    def __set_color_by_var(self) -> None:

        self.__colors = {}

        for var_nm in self.__contributions.columns:
            self.__colors[var_nm] = generate_random_color()


    def write_fig(self, name: str) -> None:

        self.__create_empty_fig()
        self.__add_plot_contributions()
        self.__add_plot_spends()
        self.__add_plot_spends_to_sales_ratio()

        self.__fig.update_layout(legend = dict(orientation = "h"))

        self.__fig.write_html("results/plot/contribution_analysis_%s.html" % name, auto_open=False)

    def __get_incremental_to_spends_ratio(self) -> Series:

        media_nms = self.__spends.columns
        total_contributions_by_media = self.__contributions[media_nms].sum(axis=0).T.sort_index()
        total_spend_by_media = self.__spends.sum(axis=0).T.sort_index()

        return total_contributions_by_media / total_spend_by_media

    def __group_at_month_level(self, df: DataFrame) -> DataFrame:

        return df.groupby(Grouper(freq="M")).sum()

    def __create_empty_fig(self) -> None:

        titles = [
            "Incremental sales by factor",
            "Incremental sales to spend ratio",
            "Spend by media"
        ]

        self.__fig = make_subplots(
            rows=2, cols=3,
            specs = [
                [{"colspan":2}, None, {"rowspan":2}],
                [{"colspan":2}, None, None]
                ],
            subplot_titles = titles
        )


    def __add_plot_contributions(self) -> None:
        
        for var_nm in self.__contributions.columns:

            self.__fig.add_trace(
                go.Bar(
                    x = self.__contributions.index,
                    y = self.__contributions[var_nm].values,
                    name = var_nm,
                    legendgroup = var_nm,
                    marker_color = self.__colors[var_nm]
                ),
                row = 1, col = 1
            )

    def __add_plot_spends(self) -> None:
        
        for media_nm in self.__spends.columns:

            self.__fig.add_trace(
                go.Bar(
                    x = self.__spends.index,
                    y = self.__spends[media_nm].values,
                    name = media_nm,
                    legendgroup = media_nm,
                    showlegend = False,
                    marker_color = self.__colors[media_nm]
                ),
                row = 2, col = 1
            )

    def __add_plot_spends_to_sales_ratio(self) -> None:

        for media_index, media_nm in enumerate(self.__incremental_to_spends_ratio.index.values):

            self.__fig.add_trace(
                go.Bar(
                    x = [""],
                    y = [self.__incremental_to_spends_ratio.values[media_index]],
                    legendgroup = media_nm,
                    name = media_nm,
                    showlegend = False,
                    marker_color = self.__colors[media_nm]
                ),
                row = 1, col = 3
            )