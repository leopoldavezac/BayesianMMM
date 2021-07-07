import pytest
import numpy as np

from bayesian_mmm.spend_transformation.carryover_visualizor import CarryoverVisualizor
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

DELAYS = [2.5, 2]
RETAIN_RATES = [0.2, 0.9]

MEDIA_NMS = ["radio", "tv"]
MAX_LAG = 3

# for geo_decay and adstock
@pytest.mark.parametrize("carryover_transfo_nm",["adstock","geo_decay"])
def test_write_fig(carryover_transfo_nm):
    
    param_nm_to_val = {"retain_rate":RETAIN_RATES}

    if carryover_transfo_nm == "adstock":
        param_nm_to_val["delay"] = DELAYS

    carryover_visualizor = CarryoverVisualizor(
        param_nm_to_val,
        MEDIA_NMS,
        MAX_LAG
    )
    carryover_visualizor.write_fig(SPENDS, "test")

    with open(OBTAINED_DIR_PATH+"carryover_test.html", "r") as f:
        obtained_html = f.read()

    file_nm = "carryover_visualization_%s.html" % carryover_transfo_nm

    with open(EXPECTED_DIR_PATH+file_nm, "r") as f:
        expected_html = f.read()

    expected_html = remove_id_ref_from_plotly_html(expected_html)
    obtained_html = remove_id_ref_from_plotly_html(obtained_html)

    expected_html = remove_color_ref_from_ploty_html(expected_html)
    obtained_html = remove_color_ref_from_ploty_html(obtained_html)

    assert expected_html == obtained_html