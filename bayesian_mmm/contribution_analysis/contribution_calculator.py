from bayesian_mmm.utilities.utilities import check_ndarray_is_matrix
from typing import Dict, Union, List
from numpy import ndarray, zeros
from pandas import DataFrame

from bayesian_mmm.inference_machine.inference_machine import InferenceMachine
from bayesian_mmm.normalizer.normalizer import Normalizer


class ContributionCalculator(InferenceMachine):

    def __init__(
        self,
        param_nm_to_val: Dict,
        max_lag: int,
        target_normalizer: Normalizer,
        media_nms: List[str],
        ctrl_nms:List[str]
        ) -> None:
        
        super().__init__(param_nm_to_val, max_lag)

        self.__target_normalizer = target_normalizer
        self.__media_nms = media_nms
        self.__ctrl_nms = ctrl_nms

    def write_results(self, name: str) -> None:

        self.__contributions.to_csv("results/contributions_%s.csv" % name, index=False)

    def get_results(self) -> None:

        return self.__contributions

    def compute_results(
            self,
            spends: ndarray,
            ctrl_vars: Union[ndarray, None],
            datetime_index: ndarray
        ) -> None:

        normalized_contributions = self.__get_normalized_results(spends, ctrl_vars)
        contributions = self.__denormalize(normalized_contributions)
        self.__contributions = self.__format(contributions, datetime_index)

    def __get_normalized_results(
        self, spends: ndarray, ctrl_vars: Union[ndarray, None]
        ) -> ndarray:

        check_ndarray_is_matrix(spends, "spends")

        if type(ctrl_vars) == ndarray:
            check_ndarray_is_matrix(ctrl_vars, "ctrl_vars")
            nb_ctrl = ctrl_vars.shape[1]
        else:
            nb_ctrl = 0

        nb_media = spends.shape[1] 
        nb_contributors = nb_media + nb_ctrl + 1 # +1 for tau aka baseline
        nb_obs = spends.shape[0]
        contributions = zeros((nb_obs, nb_contributors))

        transformed_spends = self._get_transformed_spends(spends)

        for media_index in range(nb_media):
            contributions[:, media_index] = (
                transformed_spends[:, media_index] * self._beta_medias[media_index]
            )

        for ctrl_index in range(nb_ctrl):
            contributions[:, nb_media + ctrl_index] = (
                ctrl_vars[:, ctrl_index] * self._gamma_ctrl[ctrl_index]
            )

        contributions[:, -1] = self._tau

        return contributions

    def __denormalize(self, normalized_contributions: ndarray) -> ndarray:
        
        return self.__target_normalizer.reverse_transform(normalized_contributions)

    def __format(self, contributions: ndarray, datetime_index: ndarray) -> DataFrame:

        return DataFrame(
            contributions,
            index = datetime_index,
            columns=self.__media_nms + self.__ctrl_nms + ["baseline"]
        )

