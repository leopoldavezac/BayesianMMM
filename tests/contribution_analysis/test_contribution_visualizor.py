import pytest
from pandas import DataFrame, to_datetime

from bayesian_mmm.contribution_analysis.contribution_visualizor import ContributionVisualizor
from bayesian_mmm.utilities.utilities import remove_id_ref_from_plotly_html, remove_color_ref_from_ploty_html

EXPECTED_DIR_PATH = "./tests/contribution_analysis/plot/"
OBTAINED_DIR_PATH = "./results/plot/"

contributions = DataFrame(
    [
        ["12/10/2018", 2, 3, 4, 6, 1],
        ["19/10/2018", 1, 2, 6, 7, 1],
        ["12/11/2018", 2, 3, 4, 8, 1],
        ["19/11/2018", 1, 2, 6, 9, 1]
    ],
    columns = ["week_dt", "radio", "tv", "newspaper", "consumer_index", "baseline"]
)
contributions["week_dt"] = to_datetime(contributions.week_dt, format="%d/%m/%Y")
contributions.set_index("week_dt", inplace=True)

spends = DataFrame(
    [
        ["12/10/2018", 2, 3, 4],
        ["19/10/2018", 1, 2, 6],
        ["12/11/2018", 2, 3, 4],
        ["19/11/2018", 1, 2, 6]
    ],
    columns = ["week_dt", "radio", "tv", "newspaper"]
)
spends["week_dt"] = to_datetime(spends.week_dt, format="%d/%m/%Y")
spends.set_index("week_dt", inplace=True)


def test_write_fig():

    contribution_visualizor = ContributionVisualizor(contributions, spends)
    contribution_visualizor.write_fig("test")

    with open(OBTAINED_DIR_PATH+"contribution_analysis_test.html", "r") as f:
        obtained_html = f.read()

    with open(EXPECTED_DIR_PATH+"test.html", "r") as f:
        expected_html = f.read()

    expected_html = remove_id_ref_from_plotly_html(expected_html)
    obtained_html = remove_id_ref_from_plotly_html(obtained_html)

    expected_html = remove_color_ref_from_ploty_html(expected_html)
    obtained_html = remove_color_ref_from_ploty_html(obtained_html)

    assert expected_html == obtained_html   
