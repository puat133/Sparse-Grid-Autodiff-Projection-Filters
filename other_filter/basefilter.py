from abc import ABC, abstractmethod
from typing import Callable, NamedTuple, Tuple

import jax.numpy as jnp


class FilterBase(ABC):
    """
    Base class of all stochastic filter
    """

    def __init__(self,

                 measurement_history: jnp.ndarray,
                 transition_fun: Callable,
                 output_fun: Callable,
                 constraint: Callable = None):

        self._measurement_history = measurement_history
        self._transition_fun = transition_fun
        self._output_fun = output_fun
        self._neg_log_filtering_probability = 0
        self._constraint = constraint

    @abstractmethod
    def run(self) -> Tuple:
        raise NotImplementedError

    def constrained_run(self) -> Tuple:
        pass

    @property
    def input_history(self):
        return self._input_history

    @property
    def measurement_history(self):
        return self._measurement_history

    @property
    def transition_fun(self):
        return self._transition_fun

    @property
    def output_fun(self):
        return self._output_fun

    @property
    def dynamic_parameters(self):
        return self._dynamic_parameters
