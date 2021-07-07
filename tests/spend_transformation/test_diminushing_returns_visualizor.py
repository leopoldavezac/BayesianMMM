import pytest
import numpy as np

from bayesian_mmm.spend_transformation.diminushing_returns_visualizor import DiminushingReturnsVisualizor
from bayesian_mmm.utilities.utilities import (
    remove_color_ref_from_ploty_html, remove_id_ref_from_plotly_html
)

EXPECTED_DIR_PATH = "./tests/spend_transformation/plot/"
OBTAINED_DIR_PATH = "./results/plot/"

#define delay retain rates, spends as constant
SPENDS = np.array([
    [10, 20],
    [0, 8],
    [1, 30],
    [5, 40]
])

ECS = [0.2, 1]
SLOPES = [1, 4]
HALF_SATURATIONS = [2, 3]

MEDIA_NMS = ["radio", "tv"]

# for geo_decay and adstock
@pytest.mark.parametrize("diminushing_returns_transfo_nm",["reach","hill"])
def test_write_plot(diminushing_returns_transfo_nm):
    
    param_nm_to_val = {}

    if diminushing_returns_transfo_nm == "reach":
        param_nm_to_val["half_saturation"] = HALF_SATURATIONS
    else:
        param_nm_to_val.update({
            "ec":ECS,
            "slope":SLOPES
        })

    diminushing_returns_visualizor = DiminushingReturnsVisualizor(
        param_nm_to_val,
        MEDIA_NMS
    )

    diminushing_returns_visualizor.write_fig(SPENDS, "test")

    with open(OBTAINED_DIR_PATH+"diminushing_returns_test.html", "r") as f:
        obtained_html = f.read()

    file_nm = "diminushing_returns_visualization_%s.html" % diminushing_returns_transfo_nm

    with open(EXPECTED_DIR_PATH+file_nm, "r") as f:
        expected_html = f.read()

    expected_html = remove_id_ref_from_plotly_html(expected_html)
    obtained_html = remove_id_ref_from_plotly_html(obtained_html)

    expected_html = remove_color_ref_from_ploty_html(expected_html)
    obtained_html = remove_color_ref_from_ploty_html(obtained_html)

    assert expected_html == obtained_html