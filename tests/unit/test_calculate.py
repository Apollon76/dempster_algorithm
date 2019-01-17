import os
from pathlib import Path
from typing import Tuple

import numpy as np

import pandas as pd
import pytest

from covariance_selection_algorithm import calculate


def read_csv(path: str, size: int):
    return pd.read_csv(path,
                       dtype={i: float for i in range(size)},
                       delimiter=';',
                       names=[i for i in range(size)],
                       skipinitialspace=True).values


def get_data_and_expected(name: str) -> Tuple[np.ndarray, np.ndarray]:
    cur_path = Path(os.path.dirname(os.path.abspath(__file__)))
    matrix_size = 6
    correlation_matrix = read_csv(cur_path.joinpath(f'../fixtures/{name}/data.csv'), matrix_size)
    p = correlation_matrix.shape[0]
    for i in range(p):
        for j in range(i):
            correlation_matrix[i, j] = correlation_matrix[j, i]
    expected = read_csv(cur_path.joinpath(f'../fixtures/{name}/expected.csv'), matrix_size)

    return correlation_matrix, expected


class TestCalculate:
    @pytest.mark.parametrize(
        "correlation_matrix,expected",
        [
            get_data_and_expected('dempsters_example'),
        ]
    )
    def test_dempsters_example(self,
                               correlation_matrix: np.ndarray,
                               expected: np.ndarray):
        actual = calculate(correlation_matrix, 0.5)
        assert np.allclose(expected, actual, atol=1e-4)
