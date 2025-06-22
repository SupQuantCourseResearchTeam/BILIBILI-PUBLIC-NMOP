from typing_extensions import Self
from decimal import Decimal
import numpy as np


class SequentialMeanCalculator:
    def __init__(self: Self) -> None:
        self.__mean = 0.0
        self.__n_sample = 0

    @property
    def mean(self: Self) -> float:
        return self.__mean

    @property
    def n_sample(self: Self) -> int:
        return self.__n_sample

    def step(self: Self, data_point: float) -> None:
        weight = self.__n_sample / (self.__n_sample + 1)
        self.__mean = weight * self.__mean + (1.0 - weight) * data_point
        self.__n_sample += 1


def get_relative_error(benchmark: float, numerical_value: float, precision: int = 4) -> float:
    return str(round(Decimal(100 * abs(numerical_value / benchmark - 1)), precision))


def generate_tridiagonal_matrix(
    neg: np.ndarray,
    main: np.ndarray,
    pos: np.ndarray,
) -> np.ndarray:
    if len(neg.shape) != 1:
        raise ValueError(
            f"Arg 'neg' must be 1-dimensional, but {len(neg.shape)} dimensions are detected."
        )
    if len(main.shape) != 1:
        raise ValueError(
            f"Arg 'main' must be 1-dimensional, but {len(main.shape)} dimensions are detected."
        )
    if len(pos.shape) != 1:
        raise ValueError(
            f"Arg 'pos' must be 1-dimensional, but {len(pos.shape)} dimensions are detected."
        )
    if not (pos.shape[0] == main.shape[0]):
        raise ValueError(
            f"Arg 'pos' must have equal length to arg 'main': but now {pos.shape[0]} and {main.shape[0]} are detected."
        )
    if not (neg.shape[0] == main.shape[0]):
        raise ValueError(
            f"Arg 'neg' must have equal length to arg 'main': but now {neg.shape[0]} and {main.shape[0]} are detected."
        )

    # Build the required tridiagonal matrix
    n_rows = neg.shape[0]
    results = np.concat((np.diag(neg), np.zeros((n_rows, 2))), axis=1)
    results[:, 1 : (n_rows + 1)] = results[:, 1 : (n_rows + 1)] + np.diag(main)
    results[:, 2 : (n_rows + 2)] = results[:, 2 : (n_rows + 2)] + np.diag(pos)
    return results
