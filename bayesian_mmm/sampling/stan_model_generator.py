from typing import Dict, List

from bayesian_mmm.sampling.stan_model_wrapper import StanModelWrapper

class StanModelGenerator:

    def __init__(
        self,
        carryover_transfo_nm: str,
        diminushing_returns_transfo_nm: str,
        with_ctrl_vars: bool
        ) -> None:

        self.__catch_input_error_carryover_transfo_nm(carryover_transfo_nm)
        self.__catch_input_error_diminushing_returns_transfo_nm(diminushing_returns_transfo_nm)

        self.__carryover_transfo_nm = carryover_transfo_nm
        self.__diminushing_returns_transfo_nm = diminushing_returns_transfo_nm
        self.__with_ctrl_vars = with_ctrl_vars

    def __catch_input_error_carryover_transfo_nm(self, carryover_transfo_nm: str) -> None:

        if carryover_transfo_nm not in ["adstock", "geo_decay"]:
            raise ValueError("carryover_transfo_nm must be 'adstock' or 'geo_decay'")

    def __catch_input_error_diminushing_returns_transfo_nm(
        self, diminushing_returns_transfo_nm: str
        ) -> None:

            if diminushing_returns_transfo_nm not in ["hill", "reach"]:
                raise ValueError("diminushing_returns_transfo_nm must be 'hill' or 'reach'")

    def create_model(self) -> None:

        self.__create_functions_code()
        self.__create_data_code()
        self.__create_parameters_code()
        self.__create_transformed_parameters_code()
        self.__create_model_code()
        self.__compile_code()

    def get_model(self) -> StanModelWrapper:

        try:
            return self.__model
        except AttributeError:
            print("Model is not created yet")


    def __create_functions_code(self) -> None:

        self.__function_code = "functions {\n %s %s\n}"
        
        carryover_code = self.__get_carryover_code()
        diminushing_returns_code = self.__get_diminushing_returns_code()

        self.__function_code = self.__function_code % (
            diminushing_returns_code, carryover_code
            )
        

    def __get_diminushing_returns_code(self) -> str:

        if self.__diminushing_returns_transfo_nm == "hill":
            code =  (
                "real Hill(real t, real ec, real slope) {\n"
                "   return 1 / (1 + (t / ec)^(-slope));\n"
                "}\n"
            )
        elif self.__diminushing_returns_transfo_nm == "reach":
            code =  (
                "real Reach(real t, real half_saturation) {\n"
                "   return (1 - exp(-half_saturation*t)) / (1 + exp(-half_saturation*t));\n"
                "}\n"
                )

        return code

    def __get_carryover_code(self) -> str:

        if self.__carryover_transfo_nm == "geo_decay":
            code = (
                "real Geo_decay(row_vector t, int max_lag, real retain_rate) {\n"
                "   row_vector[max_lag] weights;\n"
                "   for (lag in 1 : max_lag) {\n"
                "       weights[lag] <- pow(retain_rate, lag - 1);\n"
                "   }\n"
                "   return dot_product(t, weights) / sum(weights);\n"
                "}"
            )
        elif self.__carryover_transfo_nm == "adstock":
            code = (
                "real Adstock(row_vector t, int max_lag, real retain_rate, real delay) {\n"
                "   row_vector[max_lag] weights;\n"
                "   for (lag in 1 : max_lag) {\n"
                "       weights[lag] <- pow(retain_rate, (lag - 1 - delay) ^ 2);\n"
                "   }\n"
                "   return dot_product(t, weights) / sum(weights);\n"
                "}"
            )

        return code

    def __create_data_code(self) -> None:

        self.__data_code = (
            "data {\n"
            "   int<lower=1> N;\n"
            "   real<lower=0> Y[N];\n"
            "   int<lower=1> max_lag;\n"
            "   int<lower=1> num_media;\n"
            "   row_vector[max_lag] X_media[N, num_media];\n"
            "%s}"
        )

        if self.__with_ctrl_vars:

            self.__data_code = self.__data_code % (
                "   int<lower=1> num_ctrl;\n"
                "   row_vector[num_ctrl] X_ctrl[N];\n"
            )

        else:

            self.__data_code = self.__data_code % ""


    def __create_parameters_code(self) -> None:

        self.__parameters_code = (
            "parameters {\n"
            "   real<lower=0> noise_var;\n"
            "   real<lower=0> tau;\n"
            "   vector<lower=0>[num_media] beta_medias;\n%s%s%s"
            "}"
        )

        carryover_parameters = self.__get_carryover_parameters()
        ctrl_vars_parameters = self.__get_ctrl_vars_parameters()
        diminushing_returns_parameters = self.__get_diminushing_returns_parameters()

        self.__parameters_code = self.__parameters_code % (
            carryover_parameters, ctrl_vars_parameters, diminushing_returns_parameters
        )

    def __get_carryover_parameters(self) -> str:

        if self.__carryover_transfo_nm == "geo_decay":
            code = (
                "    vector<lower=0, upper=1>[num_media] retain_rate;\n"
            )
        elif self.__carryover_transfo_nm == "adstock":
            code = (
                "   vector<lower=0, upper=1>[num_media] retain_rate;\n"
                "   vector<lower=0,upper=max_lag-1>[num_media] delay;\n"
            )

        return code

    def __get_ctrl_vars_parameters(self) -> str:

        if self.__with_ctrl_vars:
            code = "    vector[num_ctrl] gamma_ctrl;\n"
        else:
            code = ""

        return code

    def __get_diminushing_returns_parameters(self) -> str:

        if self.__diminushing_returns_transfo_nm == "hill":
            code = (
                "   vector<lower=0, upper=1>[num_media] ec;\n"
                "   vector<lower=0>[num_media] slope;\n"
            )
        elif self.__diminushing_returns_transfo_nm == "reach":
            code = (
                "   vector<lower=0>[num_media] half_saturation;\n"
            )

        return code


    def __create_transformed_parameters_code(self) -> None:

        self.__transformed_parameters_code = (
            "transformed parameters {\n"
            "   real mu[N];\n"
            "   real cum_effect;\n"
            "   row_vector[num_media] cum_effects_hill[N];\n"
            "   for (nn in 1:N) {\n"
            "       for (media in 1 : num_media) {\n"
            "           cum_effect <- %s\n"
            "           cum_effects_hill[nn, media] <- %s\n"
            "       }\n"
            "    mu[nn] <- tau +\n"
            "    dot_product(cum_effects_hill[nn], beta_medias)\n"
            "    %s;\n"
            "   }\n"
            "}"
        )

        carryover_call = self.__get_carryover_call()
        diminushing_returns_call = self.__get_diminushing_returns_call()

        if self.__with_ctrl_vars:
            ctrl_contribution = "+dot_product(X_ctrl[nn], gamma_ctrl)"
        else:
            ctrl_contribution = ""

        self.__transformed_parameters_code = self.__transformed_parameters_code % (
            carryover_call, 
            diminushing_returns_call,
            ctrl_contribution
        )

    def __get_carryover_call(self) -> str:

        if self.__carryover_transfo_nm == "adstock":
            call = "Adstock(X_media[nn, media], max_lag, retain_rate[media], delay[media]);"
        elif self.__carryover_transfo_nm == "geo_decay":
            call = "Geo_decay(X_media[nn, media], max_lag, retain_rate[media]);"

        return call

    def __get_diminushing_returns_call(self) -> str:

        if self.__diminushing_returns_transfo_nm == "hill":
            call = "Hill(cum_effect, ec[media], slope[media]);"
        elif self.__diminushing_returns_transfo_nm == "reach":
            call = "Reach(cum_effect, half_saturation[media]);"

        return call

    def __create_model_code(self) -> None:

        self.__model_code = (
            "model {\n"
            "%s"
            "%s"
            "   tau ~ normal(0,5);\n"
            "   for (media_index in 1 : num_media) {\n"
            "       beta_medias[media_index] ~ normal(0,1);\n"
            "   }\n"
            "%s"
            "   noise_var ~ inv_gamma(0.05, 0.05 * 0.01);\n"
            "   Y ~ normal(mu, sqrt(noise_var));\n"
            "}"
        )

        carryover_prior = self.__get_carryover_prior()
        diminushing_returns_prior = self.__get_diminushing_returns_prior()
        ctrl_coef_prior = self.__get_ctrl_coef_prior()

        self.__model_code = self.__model_code % (
            carryover_prior,
            diminushing_returns_prior,
            ctrl_coef_prior
        )

    def __get_carryover_prior(self) -> str:

        if self.__carryover_transfo_nm == "adstock":
            prior = (
                "   retain_rate ~ beta(3,3);\n"
                "   delay ~ uniform(0, max_lag - 1);\n"
            )
        elif self.__carryover_transfo_nm == "geo_decay":
            prior = (
                "   retain_rate ~ beta(3,3);\n"
            )

        return prior

    def __get_diminushing_returns_prior(self) -> str:

        if self.__diminushing_returns_transfo_nm == "hill":
            prior = (
                "   slope ~ gamma(3,1);\n"
                "   ec ~ beta(2,2);\n"
            )
        elif self.__diminushing_returns_transfo_nm == "reach":
            prior = (
                "   half_saturation ~ gamma(3,1);\n"
            )
        
        return prior

    def __get_ctrl_coef_prior(self) -> str:

        if self.__with_ctrl_vars:
            prior = (
                "   for (ctrl_index in 1 : num_ctrl){\n"
                "       gamma_ctrl[ctrl_index] ~ normal(0,1);\n"
                "   }\n"
            )
        else:
            prior = ""

        return prior

    def __compile_code(self) -> None:
        
        code = (
            self.__function_code + '\n'
            + self.__data_code + '\n'
            + self.__parameters_code + '\n'
            + self.__transformed_parameters_code + '\n'
            + self.__model_code
        )

        self.__model = StanModelWrapper(code=code)
        self.__model.compile()
