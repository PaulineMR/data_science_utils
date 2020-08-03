# Standard imports
import unittest

# External imports
import pandas as pd

# Internal imports
from src.models.model_kmeans import KMeansModel
from src.models.exceptions import NotFittedError


class TestMultilabelClassification(unittest.TestCase):
    def setUp(self):
        d = [
            {"v1": 1, "v2": 1},
            {"v1": 3, "v2": 2},
            {"v1": 24, "v2": 62},
            {"v1": 28, "v2": 61},
        ]
        self.df = pd.DataFrame(d, index=range(len(d)))

    def tearDown(self):
        pass

    def test_KMeansModel(self):
        df_data = self.df.copy()
        # If no optional value has been changed, the ClassifierChainModel should have the default values
        kmeans = KMeansModel()
        self.assertEqual(kmeans.k, 2)
        self.assertEqual(kmeans.max_nb_iterations, 100)
        self.assertFalse(kmeans.trained())
        # The KMeans should be trained after being fitted and it should have k centroids
        kmeans.fit(df_data)
        self.assertTrue(kmeans.trained())
        self.assertEqual(len(kmeans._centroids), kmeans.k)
        # The output of KMeans.predict should be a list of int of the same size of the input
        prediction = kmeans.predict(df_data)
        self.assertIsInstance(prediction, list)
        self.assertEqual(len(prediction), len(df_data))
        # The clusters obtained should group the data correctly
        self.assertEqual(prediction[0], prediction[1])
        self.assertEqual(prediction[2], prediction[3])

    def test_KMeansModel_Exception(self):
        # Should raise an error if the parameters are not positive integers
        try:
            KMeansModel(k=-10)
        except TypeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        try:
            KMeansModel(k="hello")
        except TypeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        try:
            KMeansModel(max_nb_iterations=-10)
        except TypeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
        try:
            KMeansModel(max_nb_iterations="hello")
        except TypeError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

        # Should raise an error if using KMeansModel.predict before KMeansModel.fit
        kmeans = KMeansModel()
        try:
            kmeans.predict(self.df)
        except NotFittedError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)
