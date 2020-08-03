# Standard imports
import unittest

# External imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Internal imports
from src.models.multilabel_classification import (
    series_of_list_to_dataframe,
    unique_value_series_of_list,
    ClassifierChainModel,
    EnsembleClassifierChainModel,
)
from src.models.exceptions import NotFittedError


class TestMultilabelClassification(unittest.TestCase):
    def setUp(self):
        self.output_feature = "movie"
        self.possible_labels_expected = ["SF", "Drama", "Romance"]
        dictionary_for_dataset = [
            {"movie": ["SF", "Drama"], "duration": 75},
            {"movie": ["Romance", "Drama"], "duration": 67},
            {"movie": ["SF"], "duration": 86},
            {"movie": ["Romance", "Drama"], "duration": 73},
        ]
        self.df = pd.DataFrame(dictionary_for_dataset, index=range(4))
        dictionary_for_prediction = [
            {"duree": 75},
            {"duree": 67},
            {"duree": 86},
            {"duree": 73},
        ]
        self.df_for_prediction = pd.DataFrame(dictionary_for_prediction, index=range(4))

    def tearDown(self):
        pass

    def test_unique_value_series_of_list(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        # The output of test_unique_value_series_of_list should be a list
        self.assertIsInstance(possible_labels, list)
        # The output of test_unique_value_series_of_list and the expected value should be the same
        possible_labels.sort()
        self.possible_labels_expected.sort()
        self.assertEqual(len(possible_labels), len(self.possible_labels_expected))
        self.assertEqual(possible_labels, self.possible_labels_expected)

    def test_series_of_list_to_dataframe(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        df_decomposed = series_of_list_to_dataframe(
            self.df, self.output_feature, possible_labels
        )
        # The output of df_decomposed should have as columns: the possible labels and the original columns minus the
        # output feature
        expected_columns = list(self.df.columns)
        expected_columns += possible_labels
        expected_columns.remove(self.output_feature)
        self.assertEqual(len(df_decomposed.columns), len(expected_columns))
        cols = list(df_decomposed.columns)
        cols.sort()
        expected_columns.sort()
        self.assertEqual(cols, expected_columns)

    def test_ClassifierChainModel(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        df_decomposed = series_of_list_to_dataframe(
            self.df, self.output_feature, possible_labels
        )
        # If no optional value has been changed, the ClassifierChainModel should have the default values
        CC = ClassifierChainModel(labels=possible_labels)
        self.assertEqual(CC.get_labels(), possible_labels)
        self.assertIsInstance(CC.get_basic_model(), DecisionTreeClassifier)
        self.assertFalse(CC.trained())
        # The ClassifierChainModel should be trained after being fitted and the number of model in the chain should be
        # the number of label
        CC.fit(df_decomposed)
        self.assertTrue(CC.trained())
        self.assertEqual(len(CC._chain_of_models), len(CC.get_labels()))
        # The output of ClassifierChainModel.predict should be a pd.DataFrame with possible_labels as columns
        prediction = CC.predict(self.df_for_prediction)
        self.assertIsInstance(prediction, pd.DataFrame)
        cols = list(prediction.columns)
        cols.sort()
        possible_labels.sort()
        self.assertEqual(cols, possible_labels)

    def test_ClassifierChainModel_Exception(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        CC = ClassifierChainModel(labels=possible_labels)
        # Should raise an error if using ClassifierChainModel.predict before ClassifierChainModel.fit
        try:
            CC.predict(self.df_for_prediction)
        except NotFittedError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_EnsembleClassifierChainModel(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        df_decomposed = series_of_list_to_dataframe(
            self.df, self.output_feature, possible_labels
        )
        # If no optional value has been changed, the EnsembleClassifierChainModel should have the values given
        ECC = EnsembleClassifierChainModel(
            labels=possible_labels, nb_estimator=10, subsample_bagging=0.5
        )
        self.assertEqual(ECC.get_labels(), possible_labels)
        self.assertIsInstance(ECC.get_basic_model(), DecisionTreeClassifier)
        self.assertEqual(ECC.get_nb_estimator(), 10)
        self.assertEqual(ECC._subsample_bagging, 0.5)
        self.assertFalse(ECC.trained())
        # The EnsembleClassifierChainModel should be trained after being fitted and the number of model in the ensemble
        # should be nb_estimator
        ECC.fit(df_decomposed)
        self.assertTrue(ECC.trained())
        self.assertEqual(len(ECC._ensemble_of_models), ECC.get_nb_estimator())
        # The output of EnsembleClassifierChainModel.predict should be a pd.DataFrame with possible_labels as columns
        prediction = ECC.predict(self.df_for_prediction)
        self.assertIsInstance(prediction, pd.DataFrame)
        cols = list(prediction.columns)
        cols.sort()
        possible_labels.sort()
        self.assertEqual(cols, possible_labels)

    def test_EnsembleClassifierChainModel_Exception(self):
        possible_labels = unique_value_series_of_list(self.df[self.output_feature])
        ECC = EnsembleClassifierChainModel(labels=possible_labels)
        # Should raise an error if using EnsembleClassifierChainModel.predict before EnsembleClassifierChainModel.fit
        try:
            ECC.predict(self.df_for_prediction)
        except NotFittedError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
