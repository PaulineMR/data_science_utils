# Standard imports
import unittest

# External imports
import pandas as pd

# Internal imports
from src.preprocessing.feature_extraction import (
    city_encoding,
    generate_periodic_variables,
    get_temporal_variables,
    get_periodic_temporal_variable,
)


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.date_feature = "date"
        self.periodic_feature = "periodic"
        self.city_feature = "city"
        dictionary_for_dataset = [
            {"periodic": 180, "date": "31/01/2020 10:00:00", "city": "Paris"},
            {"periodic": 90, "date": "01/02/2020 15:00:00", "city": "Vauxbuin"},
            {"periodic": 0, "date": "15/03/2020 00:00:00", "city": "Bordeaux"},
        ]
        self.df = pd.DataFrame(
            dictionary_for_dataset, index=range(len(dictionary_for_dataset)),
        )

    def tearDown(self):
        pass

    def test_city_encoding(self):
        df_city = city_encoding(self.df, "city")
        # The output of city_encoding should be the same lenght as the input
        self.assertEqual(len(df_city), len(self.df))
        # The output of city_encoding should have all the new columns added
        self.assertIsNotNone(df_city["CODE_POSTAL"])
        self.assertIsNotNone(df_city["SUPERFICIE"])
        self.assertIsNotNone(df_city["POPULATION"])
        self.assertIsNotNone(df_city["NB_MEN_FISC"])
        self.assertIsNotNone(df_city["MED_REVENUE"])
        self.assertIsNotNone(df_city["LATITUDE"])
        self.assertIsNotNone(df_city["LONGITUDE"])
        self.assertIsNotNone(df_city["CODE_DEPT"])
        self.assertIsNotNone(df_city["NOM_DEPT"])
        self.assertIsNotNone(df_city["NOM_REG"])
        # Check some values
        departement = list(df_city["CODE_DEPT"])
        expected_departement = ["75", "02", "33"]
        self.assertEqual(departement, expected_departement)

    def test_generate_periodic_variables(self):
        df_periodic = generate_periodic_variables(
            self.df, periodic_variable=self.periodic_feature, modulo_variable=360
        )
        # The output of generate_periodic_variables should have the cos and sin column
        cols = list(df_periodic.columns)
        self.assertIn("cos_" + self.periodic_feature, cols)
        self.assertIn("sin_" + self.periodic_feature, cols)
        self.assertNotIn(self.periodic_feature, cols)
        # The cos and sin should have the expected value
        cos = list(df_periodic["cos_" + self.periodic_feature])
        sin = list(df_periodic["sin_" + self.periodic_feature])
        expected_cos = [-1, 0, 1]
        expected_sin = [0, 1, 0]
        for i in range(len(cos)):
            self.assertAlmostEqual(cos[i], expected_cos[i])
            self.assertAlmostEqual(sin[i], expected_sin[i])

    def test_get_temporal_variables(self):
        df_temporal = self.df.copy()
        df_temporal[self.date_feature] = pd.to_datetime(
            df_temporal[self.date_feature], dayfirst=True, utc=True
        )
        df_temporal = get_temporal_variables(
            df_temporal, date_variable=self.date_feature
        )
        # The output of get_temporal_variables should have time day month and year as columns
        cols = list(df_temporal.columns)
        for feature in ["time", "day_of_week", "day_of_month", "month", "year"]:
            self.assertIn(feature, cols)
        # The new columns should have the expected value
        time = list(df_temporal["time"])
        day_of_week = list(df_temporal["day_of_week"])
        day_month = list(df_temporal["day_of_month"])
        month = list(df_temporal["month"])
        year = list(df_temporal["year"])
        expected_time = [10, 15, 0]
        expected_day_of_week = [4, 5, 6]
        expected_day_month = [31, 1, 15]
        expected_month = [1, 2, 3]
        expected_year = [2020, 2020, 2020]
        self.assertEqual(time, expected_time)
        self.assertEqual(day_of_week, expected_day_of_week)
        self.assertEqual(day_month, expected_day_month)
        self.assertEqual(month, expected_month)
        self.assertEqual(year, expected_year)

    def test_get_periodic_temporal_variable(self):
        df_temporal = self.df.copy()
        df_temporal[self.date_feature] = pd.to_datetime(
            df_temporal[self.date_feature], dayfirst=True, utc=True
        )
        df_temporal = get_periodic_temporal_variable(
            df_temporal, date_variable=self.date_feature
        )
        # The output of get_temporal_variables should have time day month and year as columns
        cols = list(df_temporal.columns)
        self.assertIn("year", cols)
        for feature in ["time", "day_of_week", "day_of_month", "month"]:
            self.assertNotIn(feature, cols)
            self.assertIn("cos_" + feature, cols)
            self.assertIn("sin_" + feature, cols)
        # Check some of the values
        # cos values
        cos_time = list(df_temporal["cos_time"])
        self.assertAlmostEqual(cos_time[2], 1)
        cos_day_month = list(df_temporal["cos_day_of_month"])
        self.assertAlmostEqual(cos_day_month[1], 1)
        cos_month = list(df_temporal["cos_month"])
        self.assertAlmostEqual(cos_month[0], 1)
        # sin values
        sin_time = list(df_temporal["sin_time"])
        self.assertAlmostEqual(sin_time[2], 0)
        sin_day_month = list(df_temporal["sin_day_of_month"])
        self.assertAlmostEqual(sin_day_month[1], 0)
        sin_month = list(df_temporal["sin_month"])
        self.assertAlmostEqual(sin_month[0], 0)
        # Year should stay the same
        year = list(df_temporal["year"])
        expected_year = [2020, 2020, 2020]
        self.assertEqual(year, expected_year)
