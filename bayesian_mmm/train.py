from bayesian_mmm.normalizer.normalizer import Normalizer
from bayesian_mmm.contribution_analysis.contribution_calculator import ContributionCalculator
from bayesian_mmm.contribution_analysis.contribution_visualizor import ContributionVisualizor
from bayesian_mmm.evaluator.evaluator import Evaluator
from bayesian_mmm.inference_machine.inference_machine import InferenceMachine, save_inference_machine
from bayesian_mmm.sampling.parameter_estimation import estimate_parameters
from bayesian_mmm.sampling.sample_visualizor import SampleVisualizor
from bayesian_mmm.sampling.sampler import Sampler
from bayesian_mmm.utilities.utilities import load_config, load_df, split_train_test
from bayesian_mmm.sampling.stan_model_generator import StanModelGenerator
from bayesian_mmm.spend_transformation.carryover_visualizor import CarryoverVisualizor
from bayesian_mmm.spend_transformation.diminushing_returns_visualizor import DiminushingReturnsVisualizor

def run(config_file_nm: str = "train") -> None:

    config = load_config(config_file_nm)

    experiment_nm = config["EXPERIMENT_NM"]

    media_nms = config["MEDIA_NMS"]
    ctrl_nms = config["CTRL_NMS"]
    target_nm = config["TARGET_NM"]
    date_nm = config["DATE_NM"]
    
    df = load_df(config["DATA_SOURCE_PATH"], date_nm)

    train, test = split_train_test(df, config["NB_TEST_OBS"])

    predictor_normalizer_args = config["NORMALIZATION"]["PREDICTORS"]
    predictor_normalizer = Normalizer(**predictor_normalizer_args)
    predictor_normalizer.fit(train[media_nms+ctrl_nms].values)
    predictor_normalizer.save("predictor_"+experiment_nm)

    train[media_nms+ctrl_nms] = predictor_normalizer.transform(
        train[media_nms+ctrl_nms].values
        )
    test[media_nms+ctrl_nms] = predictor_normalizer.transform(
        test[media_nms+ctrl_nms].values
        )

    target_normalizer_args = config["NORMALIZATION"]["TARGET"]
    target_normalizer = Normalizer(**target_normalizer_args)
    target_normalizer.fit(
        train[target_nm].values.reshape((-1,1))
    )
    target_normalizer.save("target_"+experiment_nm)

    train[target_nm] = target_normalizer.transform(
        train[target_nm].values.reshape((-1,1))
    )
    test[target_nm] = target_normalizer.transform(
        test[target_nm].values.reshape((-1,1))
    )

    stan_model_generator = StanModelGenerator(
        config["CARRYOVER_TRANSFO_NM"],
        config["DIMINUSHING_RETURNS_TRANSFO_NM"],
        len(ctrl_nms) > 0
    )
    stan_model_generator.create_model()
    stan_model = stan_model_generator.get_model()
    del stan_model_generator

    sampler = Sampler(stan_model, config["MAX_LAG"])
    sampler.create_stan_input(
        train[media_nms].values,
        train[ctrl_nms].values if len(ctrl_nms) > 0 else None,
        train[target_nm].values
    )
    sampling_results = sampler.run_sampling(
        config["SAMPLING_N_ITER"],
        config["SAMPLING_N_PROCESSORS"]
    )
    del sampler

    sample_visualizor = SampleVisualizor(
        media_nms,
        ctrl_nms if len(ctrl_nms) > 0 else None,
    )
    sample_visualizor.write_fig(sampling_results, experiment_nm)
    del sample_visualizor

    parameter_estimation = estimate_parameters(
        sampling_results, config["PARAMETER_ESTIMATOR_NM"]
        )

    carryover_visualizor = CarryoverVisualizor(
        parameter_estimation,
        media_nms,
        config["MAX_LAG"]
    )
    carryover_visualizor.write_fig(train[media_nms].values, experiment_nm)
    del carryover_visualizor

    diminushing_returns_visualizor = DiminushingReturnsVisualizor(
        parameter_estimation,
        media_nms
        )
    diminushing_returns_visualizor.write_fig(train[media_nms].values, experiment_nm)

    inference_machine = InferenceMachine(parameter_estimation, config["MAX_LAG"])
    save_inference_machine(inference_machine, experiment_nm)

    evaluator = Evaluator(
        inference_machine,
        train[media_nms].values,
        train[ctrl_nms].values if len(ctrl_nms)>0 else None,
        train[target_nm].values,
        test[media_nms].values,
        test[ctrl_nms].values if len(ctrl_nms)>0 else None,
        test[target_nm].values,
        target_normalizer,
        df.index.values
        )
    evaluator.write_performance(experiment_nm)
    evaluator.write_fig_true_vs_pred(experiment_nm)
    evaluator.save_pred(experiment_nm)

    contribution_calculator = ContributionCalculator(
        parameter_estimation,
        config["MAX_LAG"],
        target_normalizer,
        media_nms,
        ctrl_nms
    )
    contribution_calculator.compute_results(
        train[media_nms].values,
        train[ctrl_nms].values if len(ctrl_nms) > 0 else None,
        train.index.values
    )
    contribution_calculator.write_results(experiment_nm)
    
    contribution = contribution_calculator.get_results()
    contribution_visualizor = ContributionVisualizor(contribution, train[media_nms])
    contribution_visualizor.write_fig(config["EXPERIMENT_NM"])



if __name__ == "__main__":

    run()

