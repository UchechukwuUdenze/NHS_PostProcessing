import pandas as pd
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
    # def test_generate_dfs(self):
    #     actual_test = actual
    #     predicted_test = predicted 
        # path = "MESH_output_streamflow.csv"
        # df = pd.read_csv(path, skipinitialspace = True)
        # df.drop(columns=df.columns[-1], inplace = True)
        # predicted = df.iloc[:, [1, 3, 5]]
        # actual = df.iloc[:, [1, 2, 4]]
        # assert metrics.generate_dfs(csv_fpath=path) == actual_test, predicted_test

    def test_mse(self):
        assert metrics.mse(actual, predicted, [1], 182) == [2313.095496940375]

if __name__ == '__main__':
    unittest.main()