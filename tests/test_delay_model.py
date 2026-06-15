import unittest

import pandas as pd

from ready_for_takeoff.delay_model import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    add_model_features,
    modeling_frame,
)


class DelayModelTests(unittest.TestCase):
    def test_feature_columns_do_not_include_answer_key_columns(self):
        self.assertTrue(set(FEATURE_COLUMNS).isdisjoint(LEAKAGE_COLUMNS))

    def test_inbound_cascade_uses_previous_leg_arrival_delay(self):
        df = pd.DataFrame(
            {
                "date": ["2015-01-05", "2015-01-05", "2015-01-05"],
                "sched_dep_local": ["08:00", "11:00", "09:00"],
                "tail_number": ["N1", "N1", "N2"],
                "arr_delay_min": [42, 7, 99],
                "dep_hour": [8, 11, 9],
            }
        )
        featured = add_model_features(df)
        self.assertEqual(featured.loc[0, "inbound_arr_delay_min"], 0)
        self.assertEqual(featured.loc[0, "inbound_missing"], 1)
        self.assertEqual(featured.loc[1, "inbound_arr_delay_min"], 42)
        self.assertEqual(featured.loc[1, "inbound_missing"], 0)
        self.assertEqual(featured.loc[2, "inbound_arr_delay_min"], 0)

    def test_prediction_features_work_without_historical_arrival_delay(self):
        df = pd.DataFrame(
            {
                "date": ["2015-01-05"],
                "sched_dep_local": ["08:00"],
                "tail_number": ["N1"],
                "dep_hour": [8],
            }
        )
        featured = add_model_features(df)
        self.assertEqual(featured.loc[0, "inbound_arr_delay_min"], 0)
        self.assertEqual(featured.loc[0, "inbound_missing"], 1)

    def test_modeling_frame_drops_cancelled_rows_and_returns_target(self):
        df = pd.DataFrame(
            {
                "day_of_week": [1, 1],
                "dep_hour": [8, 9],
                "distance_km": [1000, 1200],
                "temp_c": [1.0, 2.0],
                "wind_speed_kmh": [10.0, 11.0],
                "wind_gust_kmh": [15.0, 16.0],
                "precip_mm": [0.0, 0.1],
                "snowfall_cm": [0.0, 0.0],
                "cloud_cover_pct": [10, 20],
                "weather_code": [0, 3],
                "inbound_arr_delay_min": [0, 30],
                "inbound_missing": [1, 0],
                "is_early_bank": [1, 0],
                "is_evening_bank": [0, 0],
                "carrier": ["AA", "AA"],
                "origin": ["ORD", "ORD"],
                "dest": ["DEN", "LAX"],
                "delayed_15": [0, 1],
                "cancelled": [0, 1],
            }
        )
        X, y = modeling_frame(df)
        self.assertEqual(len(X), 1)
        self.assertEqual(y.tolist(), [0])

if __name__ == "__main__":
    unittest.main()
