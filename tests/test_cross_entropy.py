import math
import unittest
from test import cross_entropy_loss

class TestCrossEntropyLoss(unittest.TestCase):
    def test_basic(self):
        y_pred = [0.1, 0.2, 0.7]
        y_actual = [0, 0, 1]
        expected = -sum(a * math.log(p) for a, p in zip(y_actual, y_pred))
        self.assertAlmostEqual(cross_entropy_loss(y_pred, y_actual), expected)

if __name__ == '__main__':
    unittest.main()
