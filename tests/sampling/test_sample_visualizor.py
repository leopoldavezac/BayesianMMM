import pytest
import numpy as np

from bayesian_mmm.sampling.sample_visualizor import SampleVisualizor
from bayesian_mmm.utilities.utilities import remove_id_ref_from_plotly_html

EXPECTED_DIR_PATH = "./tests/sampling/sample_plot/"
OBTAINED_DIR_PATH = "./results/plot/"


#do not use -v or -vv on pytest call for this one -> long html code comparison print = stdout overflow
@pytest.mark.parametrize(
    "carryover_transfo_nm,diminushing_returns_transfo_nm,media_nms,ctrl_nms",
    [
        ("adstock","hill",["radio", "tv"],[]),
        ("adstock","hill",["radio","tv","facebook"],["ggtrnds"]),
        ("adstock","reach",["radio","tv","facebook"],["ggtrnds","consumer_index"]),
        ("adstock","reach",["radio", "tv"],[]),
        ("geo_decay","hill",["radio", "tv"],["ggtrnds"]),
        ("geo_decay","hill",["radio","tv","facebook"],["ggtrnds","consumer_index"]),
        ("geo_decay","reach",["radio", "tv"],["ggtrnds"]),
        ("geo_decay","reach",["radio", "tv"],[])
    ]
)
def test_write_plot(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    media_nms,
    ctrl_nms
):

    nb_media = len(media_nms)
    nb_ctrl = len(ctrl_nms)

    np.random.seed(2021)

    sample = {
        "tau":np.random.rand(100),
        "beta_medias":np.random.rand(100, nb_media)
    }

    if carryover_transfo_nm == "adstock":
        sample["retain_rate"] = np.random.rand(100, nb_media)
        sample["delay"] = np.random.rand(100, nb_media)
    elif carryover_transfo_nm == "geo_decay":
        sample["retain_rate"] = np.random.rand(100, nb_media)

    if diminushing_returns_transfo_nm == "hill":
        sample["ec"] = np.random.rand(100, nb_media)
        sample["slope"] = np.random.rand(100, nb_media)
    elif diminushing_returns_transfo_nm == "reach":
        sample["half_saturation"] = np.random.rand(100, nb_media)

    if nb_ctrl > 0:
        sample["gamma_ctrl"] = np.random.rand(100, nb_ctrl)
    
    file_nm = "sample_plot_%s_%s_%s_%s.html" % (
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        "-".join(media_nms),
        "-".join(ctrl_nms)
    )

    with open(EXPECTED_DIR_PATH+file_nm, "r") as f:
        expected_html = f.read()
   
    sample_visualizor = SampleVisualizor(media_nms, ctrl_nms)
    sample_visualizor.write_fig(sample, "test")

    with open(OBTAINED_DIR_PATH+"sample_test.html", "r") as f:
        obtained_html = f.read() 

    expected_html = remove_id_ref_from_plotly_html(expected_html)
    obtained_html = remove_id_ref_from_plotly_html(obtained_html)

    assert expected_html == obtained_html   


