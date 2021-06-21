import pytest

from BayesianMMM.sampler.stan_model_generator import StanModelGenerator
from BayesianMMM.utilities.utilities import trim

# default argument as constant
CARRYOVER_TRANSFO_NM = "geo_decay"
DIMINUSHING_RETURNS_TRANSFO_NM = "hill"
WITH_CTRL_VARS = False

EXPECTED_CODE_DIR = "./BayesianMMM/tests/sampler/stan_code/"

@pytest.mark.parametrize(
    "input_nm, input_val",
    [
        ("carryover_transfo_nm", "adlock"),
        ("diminushing_returns_transfo_nm", "hil"),
    ]
)
def test_init_input_error(input_nm, input_val):

    input_nm_to_val = {
        "carryover_transfo_nm":CARRYOVER_TRANSFO_NM,
        "diminushing_returns_transfo_nm":DIMINUSHING_RETURNS_TRANSFO_NM,
        "with_ctrl_vars":WITH_CTRL_VARS
    }

    input_nm_to_val[input_nm] = input_val
    
    with pytest.raises(ValueError):
        StanModelGenerator(**input_nm_to_val)


@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm",
    [
        ("adstock", "hill"),
        ("adstock", "reach"),
        ("geo_decay", "reach"),
        ("geo_decay", "hill")
    ]
)
def test_create_function_code(carryover_transfo_nm, diminushing_returns_transfo_nm):

    code_file_nm = "expected_code_%s_%s.txt" % (
        carryover_transfo_nm, diminushing_returns_transfo_nm
    )

    with open(EXPECTED_CODE_DIR+"function/"+code_file_nm, "r") as f:
        expected_code = f.read()

    input_nm_to_val = {
        "carryover_transfo_nm":carryover_transfo_nm,
        "diminushing_returns_transfo_nm":diminushing_returns_transfo_nm,
        "with_ctrl_vars":WITH_CTRL_VARS
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    sampler._StanModelGenerator__create_functions_code()
    obtained_code = sampler._StanModelGenerator__function_code

    trim_expected_code = trim(expected_code)
    trim_obtained_code = trim(obtained_code)

    assert trim_obtained_code == trim_expected_code

@pytest.mark.parametrize("with_ctrl_vars", [True, False])
def test_create_data_code(with_ctrl_vars):

    code_file_nm = "expected_code_with_ctrl_vars_%s.txt" % str(with_ctrl_vars).lower()

    with open(EXPECTED_CODE_DIR+"data/"+code_file_nm, "r") as f:
        expected_code = f.read()

    input_nm_to_val = {
        "carryover_transfo_nm":CARRYOVER_TRANSFO_NM,
        "diminushing_returns_transfo_nm":DIMINUSHING_RETURNS_TRANSFO_NM,
        "with_ctrl_vars":with_ctrl_vars
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    sampler._StanModelGenerator__create_data_code()
    obtained_code = sampler._StanModelGenerator__data_code

    trim_expected_code = trim(expected_code)
    trim_obtained_code = trim(obtained_code)

    assert trim_obtained_code == trim_expected_code


@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars",
    [
        ("adstock", "hill", True),
        ("adstock", "hill", False),
        ("adstock", "reach", True),
        ("adstock", "reach", False),
        ("geo_decay", "reach", True),
        ("geo_decay", "reach", False),
        ("geo_decay", "hill", True),
        ("geo_decay", "hill", False)
    ]
)
def test_create_parameters_code(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars
    ):
    
    code_file_nm = "expected_code_%s_%s_with_ctrl_vars_%s.txt" % (
        carryover_transfo_nm, diminushing_returns_transfo_nm, str(with_ctrl_vars).lower()
    )
    with open(EXPECTED_CODE_DIR+"parameters/"+code_file_nm, "r") as f:
        expected_code = f.read()

    input_nm_to_val = {
        "carryover_transfo_nm":carryover_transfo_nm,
        "diminushing_returns_transfo_nm":diminushing_returns_transfo_nm,
        "with_ctrl_vars":with_ctrl_vars
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    sampler._StanModelGenerator__create_parameters_code()
    obtained_code = sampler._StanModelGenerator__parameters_code

    trim_expected_code = trim(expected_code)
    trim_obtained_code = trim(obtained_code)

    assert trim_obtained_code == trim_expected_code


@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars",
    [
        ("adstock", "hill", True),
        ("adstock", "hill", False),
        ("adstock", "reach", True),
        ("adstock", "reach", False),
        ("geo_decay", "reach", True),
        ("geo_decay", "reach", False),
        ("geo_decay", "hill", True),
        ("geo_decay", "hill", False),
    ]
)
def test_create_transformed_parameters_code(
    carryover_transfo_nm,
    diminushing_returns_transfo_nm,
    with_ctrl_vars
    ):
    
    code_file_nm = "expected_code_%s_%s_with_ctrl_vars_%s.txt" % (
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        str(with_ctrl_vars).lower()
    )
    with open(EXPECTED_CODE_DIR+"transformed_parameters/"+code_file_nm, "r") as f:
        expected_code = f.read()

    input_nm_to_val = {
        "carryover_transfo_nm":carryover_transfo_nm,
        "diminushing_returns_transfo_nm":diminushing_returns_transfo_nm,
        "with_ctrl_vars":with_ctrl_vars
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    sampler._StanModelGenerator__create_transformed_parameters_code()
    obtained_code = sampler._StanModelGenerator__transformed_parameters_code

    trim_expected_code = trim(expected_code)
    trim_obtained_code = trim(obtained_code)

    assert trim_obtained_code == trim_expected_code


@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars",
    [
        ("adstock", "hill", True),
        ("adstock", "hill", False),
        ("adstock", "reach", True),
        ("adstock", "reach", False),
        ("geo_decay", "reach", True),
        ("geo_decay", "reach", False),
        ("geo_decay", "hill", True),
        ("geo_decay", "hill", False)
    ]
)
def test_create_model_code(carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars):

    code_file_nm = "expected_code_%s_%s_with_ctrl_vars_%s.txt" % (
        carryover_transfo_nm,
        diminushing_returns_transfo_nm,
        str(with_ctrl_vars).lower()
        )

    with open(EXPECTED_CODE_DIR+"/model/"+code_file_nm, "r") as f:
        expected_code = f.read()

    input_nm_to_val = {
        "carryover_transfo_nm":carryover_transfo_nm,
        "diminushing_returns_transfo_nm":diminushing_returns_transfo_nm,
        "with_ctrl_vars":with_ctrl_vars
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    sampler._StanModelGenerator__create_model_code()
    obtained_code = sampler._StanModelGenerator__model_code

    print(expected_code, "\n\n", obtained_code)
    
    trim_expected_code = trim(expected_code)
    trim_obtained_code = trim(obtained_code)

    assert trim_obtained_code == trim_expected_code



# this one takes a long time to run
@pytest.mark.parametrize(
    "carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars",
    [
        ("adstock", "hill", True),
        ("adstock", "hill", False),
        ("adstock", "reach", True),
        ("adstock", "reach", False),
        ("geo_decay", "reach", True),
        ("geo_decay", "reach", False),
        ("geo_decay", "hill", True),
        ("geo_decay", "hill", False)
    ]
)
def test_compile_code(carryover_transfo_nm, diminushing_returns_transfo_nm, with_ctrl_vars):

    input_nm_to_val = {
        "carryover_transfo_nm":carryover_transfo_nm,
        "diminushing_returns_transfo_nm":diminushing_returns_transfo_nm,
        "with_ctrl_vars":with_ctrl_vars
    }

    sampler = StanModelGenerator(**input_nm_to_val)
    try:
        sampler.create_model()
        assert True
    except:
        assert False


# # test for np wrong dimensions
# def test_create_sampler_input_input_error():
#     pass


# # test for non np + np wrong dimensions
# def test_run_sampling_input_error():
#     pass


# # with ctrl_vars and without
# # adstock transfo is geo_decay, delayed_geo_decay
# # diminushing returns transfo is hill, reach
# def test_run_sampling():
#     # test that output is a np matrix containing sampling value
#     pass

# def test_plot_posterior():
#     # test that html code is the same as reference
#     pass

# # test catch wrong values
# def test_get_parameters_estimation_input_error():
#     pass

# # for mean, median
# def test_get_parameters_estimation():
#     pass
