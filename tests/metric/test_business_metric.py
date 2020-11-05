# Standard imports
import unittest

# Internal imports
from src.metric.business_metric import business_metric, get_important_quantiles


class TestBusinessMetric(unittest.TestCase):
    def setUp(self):
        self.truth = [1, 3, 5, 7, 9]
        self.pred = [2, 3, 7, 6, 9]
        self.representative_value = 2
        self.expected_bm = [0.5, 0, 1, 0.5, 0]

    def tearDown(self):
        pass

    def test_business_metric(self):
        bm = business_metric(self.truth, self.pred, self.representative_value)
        # The output of business_metric should be a list with the same lenght
        # as the input and it should contain expected values
        self.assertIsInstance(bm, list)
        self.assertEqual(len(bm), len(self.truth))
        for i in range(len(bm)):
            self.assertEqual(bm[i], self.expected_bm[i])

    def test_get_important_quantiles(self):
        bm = business_metric(self.truth, self.pred, self.representative_value)
        quantile_bm = get_important_quantiles(bm)
        # The output of get_quantile_business_metric should be a dict with expected values
        self.assertIsInstance(quantile_bm, dict)
        self.assertAlmostEqual(quantile_bm['0.05'], 0.)
        self.assertAlmostEqual(quantile_bm['0.25'], 0.)
