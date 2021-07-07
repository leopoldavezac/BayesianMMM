from typing import Dict, OrderedDict
import pystan


DEFAULT_CODE = """
functions {
    real Hill(real t, real ec, real slope) {
        return 1 / (1 + (t / ec)^(-slope));
    }
    real Adstock(row_vector t, row_vector weights) {
        return dot_product(t, weights) / sum(weights);
    }
}
data {
    int<lower=1> N;
    real<lower=0> Y[N];
    int<lower=1> max_lag;
    int<lower=1> num_media;
    row_vector[max_lag] X_media[N, num_media];
    int<lower=1> num_ctrl;
    row_vector[num_ctrl] X_ctrl[N];
}
parameters {
    real<lower=0> noise_var;
    real<lower=0> tau;
    vector<lower=0>[num_media] beta_medias;
    vector[num_ctrl] gamma_ctrl;
    vector<lower=0,upper=1>[num_media] retain_rate;
    vector<lower=0,upper=max_lag-1>[num_media] delay;
    vector<lower=0,upper=1>[num_media] ec;
    vector<lower=0>[num_media] slope;
}
transformed parameters {
    // a vector of the mean response
    real mu[N];
    real cum_effect;
    row_vector[num_media] cum_effects_hill[N];
    row_vector[max_lag] lag_weights;
    for (nn in 1:N) {
        for (media in 1 : num_media) {
            for (lag in 1 : max_lag) {
                lag_weights[lag] <- pow(retain_rate[media], (lag - 1 - delay[media]) ^ 2);
            }
            cum_effect <- Adstock(X_media[nn, media], lag_weights);
            cum_effects_hill[nn, media] <- Hill(cum_effect, ec[media], slope[media]);
        }
        mu[nn] <- tau +
        dot_product(cum_effects_hill[nn], beta_medias) +
        dot_product(X_ctrl[nn], gamma_ctrl);
    }
}
model {
    retain_rate ~ beta(3,3);
    delay ~ uniform(0, max_lag - 1);
    slope ~ gamma(3, 1);
    ec ~ beta(2,2);
    tau ~ normal(0, 5);
    for (media_index in 1 : num_media) {
        beta_medias[media_index] ~ normal(0, 1);
    }
    for (ctrl_index in 1 : num_ctrl) {
        gamma_ctrl[ctrl_index] ~ normal(0,1);
    }
    noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
    Y ~ normal(mu, sqrt(noise_var));
}
"""

class StanModelWrapper:

    def __init__(self, code=DEFAULT_CODE) -> None:
        
        self.__code = code

    def compile(self) -> None:

        self.__model = pystan.StanModel(model_code=self.__code, verbose=False)

    def sample(self, args: Dict, n_iter: int, chains: Dict) -> OrderedDict:

        results = self.__model.sampling(data=args, iter=n_iter, chains=chains)

        return results.extract()


