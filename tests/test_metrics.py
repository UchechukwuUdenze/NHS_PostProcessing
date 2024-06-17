import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from collections.abc import Generator
import unittest

from postprocessinglib.evaluation import metrics

path = "MESH_output_streamflow.csv"
df = pd.read_csv(path, skipinitialspace = True)
df.drop(columns=df.columns[-1], inplace = True)
predicted = df.iloc[:, [1, 3, 5]]
actual = df.iloc[:, [1, 2, 4]]

class TestLibrary(unittest.TestCase):
    def test_generate_dfs(self):
        assert_frame_equal(metrics.generate_dfs(csv_fpath=path)[0], actual)
        assert_frame_equal(metrics.generate_dfs(csv_fpath=path)[1], predicted)

    def test_mse(self):
        self.assertEqual(metrics.mse(actual, predicted, [1], 182), [2313.095496940375])

    def test_rmse(self):
        self.assertEqual(metrics.rmse(actual, predicted, [1], 182), [48.09465143797567])

    def test_mae(self):
        self.assertEqual(metrics.mae(actual, predicted, [1], 182), [25.447288581268012])

    def test_nse(self):
        self.assertEqual(metrics.nse(actual, predicted, [1], 182), [-0.004971126241519741])

    def test_kge(self):
        self.assertEqual(metrics.kge(actual, predicted, [1], 182), [0.4799990974685058])

    def test_bias(self):
        self.assertEqual(metrics.bias(actual, predicted, [1], 182), [0.286022121992192])

    def test_available_metrics(self):
        self.assertEqual(metrics.available_metrics(), ["MSE", "RMSE", "MAE", "NSE", "KGE", "PBIAS"])

    def test_calculate_all_metrics(self):
        result = {'MSE': [2313.095496940375], 'RMSE': [48.09465143797567], 'MAE': [25.447288581268012],
                   'NSE': [-0.004971126241519741], 'KGE': [0.4799990974685058], 'BIAS': [0.286022121992192]}
        self.assertEqual(metrics.calculate_all_metrics(actual, predicted, [1], 182), result)

if __name__ == '__main__':
    unittest.main()