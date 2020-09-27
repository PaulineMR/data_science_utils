# Standard imports
import unittest

# Internal imports
from src.metric.business_metric import business_metric


class TestEncoding(unittest.TestCase):
    def setUp(self):
        self.truth = [1, 3, 5, 7, 9]
        self.pred = [2, 3, 7, 6, 9]
        self.representative_value = 2
        self.expected_bm = [0.5, 0, 1, 0.5, 0]

    def tearDown(self):
        pass

    def test_ordinal_encoding(self):
        bm = business_metric(self.truth, self.pred, self.representative_value)

        for i in range(len(bm)):
            self.assertEqual(bm[i], self.expected_bm[i])