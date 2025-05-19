import math
import unittest
from test import cross_entropy_loss

class TestCrossEntropyLossEdgeCases(unittest.TestCase):
    def test_uniform_distribution(self):
        y_pred = [0.25, 0.25, 0.25, 0.25]
        y_actual = [0, 1, 0, 0]
        expected = -math.log(0.25)
        self.assertAlmostEqual(cross_entropy_loss(y_pred, y_actual), expected)

    def test_near_perfect_prediction(self):
        eps = 1e-8
        y_pred = [eps, eps, 1 - 2 * eps]
        y_actual = [0, 0, 1]
        expected = -math.log(1 - 2 * eps)
        self.assertAlmostEqual(cross_entropy_loss(y_pred, y_actual), expected)

if __name__ == '__main__':
    unittest.main()
