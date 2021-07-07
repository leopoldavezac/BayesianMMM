from typing import Union
from numpy import ndarray, abs, mean, concatenate
from json import dump
from pandas import Series
from pandas.core.frame import DataFrame
from plotly.graph_objects import Figure, Scatter

from bayesian_mmm.inference_machine.inference_machine import InferenceMachine, save_inference_machine
from bayesian_mmm.normalizer.normalizer import Normalizer

class Evaluator:

    def __init__(
        self,
        inference_machine: InferenceMachine,
        train_spends: ndarray,
        train_ctrl_vars: Union[ndarray, None],
        train_y: ndarray,
        test_spends: ndarray,
        test_ctrl_vars: Union[ndarray, None],
        test_y: ndarray,
        target_normalizer: Normalizer,
        datetime_index: ndarray
        ) -> None:

        self.__inference_machine = inference_machine
        
        self.__datetime_index = Series(datetime_index)

        self.__train_y = target_normalizer.reverse_transform(
            train_y.reshape((-1,1))
            ).reshape((-1,))
        self.__test_y = target_normalizer.reverse_transform(
            test_y.reshape((-1,1))
            ).reshape((-1,))

        self.__train_pred = self.__inference_machine.predict(
            train_spends, train_ctrl_vars
            )
        self.__train_pred = target_normalizer.reverse_transform(
            self.__train_pred.reshape((-1,1))
            ).reshape((-1,))

        self.__test_pred = self.__inference_machine.predict(
            test_spends, test_ctrl_vars
            )
        self.__test_pred = target_normalizer.reverse_transform(
            self.__test_pred.reshape((-1,1))
            ).reshape((-1,))

    def write_performance(self, name: str) -> None:

        self.__compute_performance()

        with open("./results/performance_%s.json" % name, "w") as f:
            dump(self.__performance, f)


    def __compute_performance(self) -> None:

        train_mape = self.__compute_mape(self.__train_y, self.__train_pred)
        test_mape = self.__compute_mape(self.__test_y, self.__test_pred)

        self.__performance = {
            "train": train_mape,
            "test": test_mape
        }
        
    def __compute_mape(self, true: ndarray, pred: ndarray) -> float:

        return mean(abs(true - pred) / true) * 100

    def save_pred(self, name: str) -> None:

        pred = concatenate([self.__train_pred, self.__test_pred])
        true = concatenate([self.__train_y, self.__test_y])

        pred = pred.reshape((-1,1))
        true = true.reshape((-1,1))

        pred_vs_true = DataFrame(
            concatenate([pred, true], axis=1),
            columns=["pred", "true"],
            index = self.__datetime_index
            )
        pred_vs_true.index.rename("dt", inplace=True)
            
        pred_vs_true.to_csv("./results/prediction_%s.csv" % name)

    def write_fig_true_vs_pred(self, name: str) -> None:

        pred = concatenate([self.__train_pred, self.__test_pred])
        true = concatenate([self.__train_y, self.__test_y])

        fig = Figure()

        fig.add_trace(
            Scatter(
                x = self.__datetime_index,
                y = true,
                name = "true"
            )
        )
        fig.add_trace(
            Scatter(
                x = self.__datetime_index,
                y = pred,
                name = "pred"
            )
        )

        fig.write_html("./results/plot/true_vs_pred_%s.html" % name, auto_open=False)



