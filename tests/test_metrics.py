import pandas as pd
from pandas.testing import assert_frame_equal
import unittest

from postprocessinglib.evaluation import metrics
from postprocessinglib.utilities.errors import AllInvalidError

path = "C:/Users/udenzeU/OneDrive - EC-EC/Fuad_Mesh_Dataset/MESH-bow-at-banff/results/MESH_output_streamflow.csv"
df = pd.read_csv(path, skipinitialspace = True, index_col = ["YEAR", "JDAY"])
df.drop(columns=df.columns[-1], inplace = True)
warm_up = 365
actual = df.iloc[warm_up:, [0, 2]]
predicted = df.iloc[warm_up:, [1, 3]]
stations = df.iloc[warm_up:, [0, 1]]

class TestLibrary(unittest.TestCase):
    def test_generate_dataframes(self):
        assert_frame_equal(metrics.generate_dataframes(csv_fpath=path, num_min=warm_up)[0], actual)
        assert_frame_equal(metrics.generate_dataframes(csv_fpath=path, num_min=warm_up)[1], predicted)

    def test_station_dataframe(self):
        for station in metrics.station_dataframe(actual, predicted, 1):
            assert_frame_equal(station, stations)

    def test_mse(self):
        self.assertEqual(metrics.mse(actual, predicted, 1), [1656.685638835447])

    def test_rmse(self):
        self.assertEqual(metrics.rmse(actual, predicted, 1), [40.702403354537275])

    def test_mae(self):
        self.assertEqual(metrics.mae(actual, predicted, 1), [22.12912878335626])

    def test_nse(self):
        self.assertEqual(metrics.nse(actual, predicted, 1), [0.0021806971124596064])

    def test_kge_2009(self):
        self.assertEqual(metrics.kge_2009(actual, predicted, 1), [0.49061085454963205])

    def test_kge_2012(self):
        self.assertEqual(metrics.kge_2012(actual, predicted, 1), [0.27812840065858213])

    def test_bias(self):
        self.assertEqual(metrics.bias(actual, predicted, 1), [-27.052012466427488])

    def test_available_metrics(self):
        self.assertEqual(metrics.available_metrics(), ["MSE", "RMSE", "MAE", "NSE", "KGE 2009", "KGE 2012", "PBIAS"])

    def test_calculate_all_metrics(self):
        result = {'MSE': [1656.685638835447], 'RMSE': [40.702403354537275], 'MAE': [22.12912878335626],
                   'NSE': [0.0021806971124596064], 'KGE 2009': [0.49061085454963205], 'KGE 2012': [0.27812840065858213], 
                   'BIAS': [-27.052012466427488]}
        self.assertEqual(metrics.calculate_all_metrics(actual, predicted, 1), result)

    def test_calculate_metrics(self):
        check_metrices = ["MSE", "RMSE", "MAE", "NSE", "KGE 2009", "KGE 2012", "PBIAS"]
        result = {'MSE': [1656.685638835447], 'RMSE': [40.702403354537275], 'MAE': [22.12912878335626],
                   'NSE': [0.0021806971124596064], 'KGE 2009': [0.49061085454963205], 'KGE 2012': [0.27812840065858213], 
                   'BIAS': [-27.052012466427488]}
        self.assertEqual(metrics.calculate_metrics(actual, predicted, check_metrices, 1), result)

    def test_check_valid_dataframe(self):
        try:
            metrics.check_valid_dataframe(metrics.generate_dataframes(path, warm_up)[0],
                                          metrics.generate_dataframes(path, warm_up)[1])
        except AllInvalidError:
            self.fail("Test for invalid values failed")

    def test_validate_data(self):
        try: 
            metrics.validate_data(metrics.generate_dataframes(path, warm_up)[0], 
                                  metrics.generate_dataframes(path, warm_up)[1])
        except RuntimeError:
            self.fail("Incomplete data or wrong shape")
        except ValueError:
            self.fail("One of them isn't a Dataframe")
    
    def test_filter_valid_data(self):
        self.assertEqual(len(metrics.filter_valid_data(actual, station_num=0, neg = 1)) , 5816)

if __name__ == '__main__':
    unittest.main()