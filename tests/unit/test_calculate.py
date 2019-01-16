import os
from pathlib import Path
from typing import Tuple

import numpy as np

import pandas as pd
import pytest

from covariance_selection_algorithm import calculate


def get_data_and_expected(name: str) -> Tuple[np.ndarray, np.ndarray]:
    cur_path = Path(os.path.dirname(os.path.abspath(__file__)))
    correlation_matrix = pd.read_csv(cur_path.joinpath(f'../fixtures/{name}/data.csv'),
                                     dtype={i: float for i in range(6)},
                                     delimiter=';',
                                     names=[i for i in range(6)],
                                     skipinitialspace=True).values
    p = correlation_matrix.shape[0]
    for i in range(p):
        for j in range(i):
            correlation_matrix[i, j] = correlation_matrix[j, i]

    expected = pd.read_csv(cur_path.joinpath(f'../fixtures/{name}/expected.csv'),
                           dtype={i: float for i in range(6)},
                           delimiter=';',
                           names=[i for i in range(6)],
                           skipinitialspace=True).values

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
