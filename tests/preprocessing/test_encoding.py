# Standard imports
import unittest

# External imports
import pandas as pd

# Internal imports
from src.preprocessing.encoding import (
    ordinal_encoding,
    one_hot_encoding,
    target_encoding,
)


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.output_feature_classification = "size"
        self.labels = ["small", "medium", "big"]
        dictionary_for_dataset_classification = [
            {"size": "small", "random": 42},
            {"size": "medium", "random": 42},
            {"size": "small", "random": 42},
            {"size": "small", "random": 42},
            {"size": "big", "random": 42},
            {"size": "medium", "random": 42},
        ]
        self.df_classification = pd.DataFrame(
            dictionary_for_dataset_classification,
            index=range(len(dictionary_for_dataset_classification)),
        )

        self.output_feature_regression = "size"
        self.categorical_feature = "animal"
        dictionary_for_dataset_regression = [
            {"size": 30, "animal": "cat"},
            {"size": 9, "animal": "hamster"},
            {"size": 25, "animal": "cat"},
            {"size": 24, "animal": "cat"},
            {"size": 46, "animal": "dog"},
            {"size": 10, "animal": "hamster"},
        ]
        self.df_regression = pd.DataFrame(
            dictionary_for_dataset_regression,
            index=range(len(dictionary_for_dataset_regression)),
        )

    def tearDown(self):
        pass

    def test_ordinal_encoding(self):
        df_ordinal = ordinal_encoding(
            self.df_classification,
            categorical_variable=self.output_feature_classification,
        )
        # The output of ordinal_encoding should be a pd.DataFrame with the same columns as the original DataFrame and
        # the output feature contains only number
        cols_ordinal = list(df_ordinal.columns)
        cols_ordinal.sort()
        cols_original = list(self.df_classification.columns)
        cols_original.sort()
        self.assertEqual(cols_ordinal, cols_original)
        possible_values = list(df_ordinal[self.output_feature_classification].unique())
        possible_values.sort()
        self.assertEqual(possible_values, list(range(len(self.labels))))

    def test_one_hot_encoding(self):
        df_one_hot = one_hot_encoding(
            self.df_classification,
            categorical_variable=self.output_feature_classification,
            prefix_new_feature="",
        )
        # The output of one_hot_encoding should have as columns: the possible values and the original columns minus the
        # output feature
        expected_columns = list(self.df_classification.columns)
        expected_columns += self.labels
        expected_columns.remove(self.output_feature_classification)
        expected_columns = list(set(expected_columns))
        cols = list(df_one_hot.columns)
        cols.sort()
        expected_columns.sort()
        self.assertEqual(cols, expected_columns)

    def test_target_encoding(self):
        df_target = target_encoding(
            self.df_regression,
            target_variable=self.output_feature_regression,
            categorical_variable=self.categorical_feature,
        )
        # The output of target_encoding should be a pd.DataFrame with the same columns as the original DataFrame and
        # the output feature contains only number
        cols_target = list(df_target.columns)
        cols_target.sort()
        cols_original = list(self.df_regression.columns)
        cols_original.sort()
        self.assertEqual(cols_target, cols_original)
        values = df_target[self.output_feature_regression]
        self.assertIsInstance(sum(values), (int, float))
