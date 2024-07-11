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
    maxDiff = None
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

    def test_lognse(self):
        self.assertEqual(metrics.lognse(actual, predicted, 1), [-0.23787575482558498])

    def test_kge(self):
        self.assertEqual(metrics.kge(actual, predicted, 1), [0.49061085454963205])

    def test_kge_2012(self):
        self.assertEqual(metrics.kge_2012(actual, predicted, 1), [0.27812840065858213])

    def test_bias(self):
        self.assertEqual(metrics.bias(actual, predicted, 1), [-27.052012466427488])

    def test_TTP(self):
        self.assertEqual(metrics.time_to_peak(actual, 1), [167.375])

    def test_TTCoM(self):
        self.assertEqual(metrics.time_to_centre_of_mass(actual, 1), [193.911419943451])

    def test_SPOD(self):
        self.assertEqual(metrics.SpringPulseOnset(actual, 1), [136.875])

    def test_available_metrics(self):
        self.assertEqual(metrics.available_metrics(), [
        "MSE", "RMSE", "MAE", "NSE", "NegNSE", "LogNSE", "NegLogNSE",
        "KGE", "NegKGE", "KGE 2012", "BIAS", "AbsBIAS", "TTP", "TTCoM", "SPOD" 
    ])

    def test_calculate_all_metrics(self):
        result = {'MSE': [1656.685638835447, 730.4265136887851],
                  'RMSE': [40.702403354537275, 27.02640400957525],
                  'MAE': [22.12912878335626, 13.558496468765133],
                  'NSE': [0.0021806971124596064, -4.824700963459696],
                  'NegNSE': [-0.0021806971124596064, 4.824700963459696],
                  'LogNSE': [-0.23787575482558498, -0.8608412562580361],
                  'NegLogNSE': [0.23787575482558498, 0.8608412562580361],
                  'KGE': [0.49061085454963205, -0.7628035230858525],
                  'NegKGE': [-0.49061085454963205, 0.7628035230858525],
                  'KGE 2012': [0.27812840065858213, -0.1908546908670441],
                  'BIAS': [-27.052012466427488, 31.551597410264947],
                  'AbsBIAS': [27.052012466427488, 31.551597410264947],
                  'TTP_obs': [167.375, 170.77777777777777],
                  'TTP_sim': [186.61111111111111, 172.72222222222223],
                  'TTCoM_obs': [193.911419943451, 203.2721509619546],
                  'TTCoM_sim': [192.78635358563776, 192.42788801235076],
                  'SPOD_obs': [136.875, 134.72222222222223],
                  'SPOD_sim': [136.19444444444446, 142.94444444444446]}
        self.assertEqual(metrics.calculate_all_metrics(actual, predicted, 2), result)

    def test_calculate_metrics(self):
        check_metrices = ["MSE", "RMSE", "MAE", "NSE", "NegNSE", "LogNSE", "NegLogNSE",
                          "KGE", "NegKGE", "KGE 2012", "BIAS", "AbsBIAS", "TTP_obs",
                          "TTP_sim", "TTCoM_obs", "TTCoM_sim", "SPOD_obs", "SPOD_sim"]
        result = {'MSE': [1656.685638835447, 730.4265136887851],
                    'RMSE': [40.702403354537275, 27.02640400957525],
                    'MAE': [22.12912878335626, 13.558496468765133],
                    'NSE': [0.0021806971124596064, -4.824700963459696],
                    'NegNSE': [-0.0021806971124596064, 4.824700963459696],
                    'LogNSE': [-0.23787575482558498, -0.8608412562580361],
                    'NegLogNSE': [0.23787575482558498, 0.8608412562580361],
                    'KGE': [0.49061085454963205, -0.7628035230858525],
                    'NegKGE': [-0.49061085454963205, 0.7628035230858525],
                    'KGE 2012': [0.27812840065858213, -0.1908546908670441],
                    'BIAS': [-27.052012466427488, 31.551597410264947],
                    'AbsBIAS': ([27.052012466427488, 31.551597410264947],),
                    'TTP_obs': [167.375, 170.77777777777777],
                    'TTP_sim': [186.61111111111111, 172.72222222222223],
                    'TTCoM_obs': [193.911419943451, 203.2721509619546],
                    'TTCoM_sim': [192.78635358563776, 192.42788801235076],
                    'SPOD_obs': [136.875, 134.72222222222223],
                    'SPOD_sim': [136.19444444444446, 142.94444444444446]}
        self.assertEqual(metrics.calculate_metrics(actual, predicted, check_metrices, 2), result)

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

    def test_is_leap_year(self):
        self.assertEqual(metrics.is_leap_year(2020), True)
        self.assertEqual(metrics.is_leap_year(2021), False)

if __name__ == '__main__':
    unittest.main()