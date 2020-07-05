import unittest
import numpy as np
from proflogit.empc import EMPChurn


class TestEMPC(unittest.TestCase):
    # Class variable
    y_true = [1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
    y_score = [
        0.6125478,
        0.3642710,
        0.4321361,
        0.1402911,
        0.3848959,
        0.2444155,
        0.9706413,
        0.8901728,
        0.7817814,
        0.8687518,
    ]
    empc_test_val = 30.3003222629848
    empc_params = {
        "alpha": 6,
        "beta": 14,
        "clv": 200,
        "d": 10,
        "f": 1,
        "case_label": 1,
    }

    def test_empc_y_score_instantiation(self):
        """Test EMPC is computed correctly: y_score given at instantiation."""
        test_val = self.empc_test_val
        y = self.y_true
        yhat = self.y_score
        params = self.empc_params
        emp = EMPChurn(y_true=y, y_score=yhat, **params)
        empc = emp.empc()
        self.assertAlmostEqual(empc, test_val)

    def test_empc_y_score_method(self):
        """Test EMPC is computed correctly: y_score given as method input."""
        test_val = self.empc_test_val
        y = self.y_true
        yhat = self.y_score
        params = self.empc_params
        emp = EMPChurn(y_true=y, **params)
        empc = emp.empc(y_score=yhat)
        self.assertAlmostEqual(empc, test_val)

    def test_direct_empc_computation(self):
        """Test EMPC is computed correctly: Immediately accessing method."""
        test_val = self.empc_test_val
        y = self.y_true
        yhat = self.y_score
        params = self.empc_params
        empc1 = EMPChurn(y_true=y, y_score=yhat, **params).empc()
        empc2 = EMPChurn(y_true=y, **params).empc(y_score=yhat)
        self.assertAlmostEqual(empc1, test_val)
        self.assertAlmostEqual(empc2, test_val)

    def test_repeated_empc_computation(self):
        """
        Test EMPC is computed correctly: Repeated computation with same object.
        """
        test_val = 30.3003222629848
        y = np.asarray(self.y_true)
        yhat = np.array(self.y_score)
        params = self.empc_params

        obj = EMPChurn(y_true=y, **params)
        empc1 = obj.empc(y_score=yhat)
        self.assertAlmostEqual(empc1, test_val)

        empc2 = obj.empc()
        self.assertAlmostEqual(empc2, test_val)

        yhat[0] = 0.287577520124614
        empc3 = obj.empc(y_score=yhat)
        self.assertAlmostEqual(empc3, 30.3001182343737)

        yhat[6] = 0.113703411305323
        empc4 = obj.empc(y_score=yhat)
        self.assertAlmostEqual(empc4, 29.2002764792561)

    def test_y_true_labels_str(self):
        test_val = self.empc_test_val
        y = ["yes" if y_lbl == 1 else "no" for y_lbl in self.y_true]
        yhat = self.y_score
        params = self.empc_params
        params["case_label"] = "yes"
        emp = EMPChurn(y_true=y, y_score=yhat, **params)
        empc = emp.empc()
        self.assertAlmostEqual(empc, test_val)

    def test_y_true_labels_float(self):
        test_val = self.empc_test_val
        y = [float(l) for l in self.y_true]
        yhat = self.y_score
        params = self.empc_params
        emp = EMPChurn(y_true=y, y_score=yhat, **params)
        empc = emp.empc()
        self.assertAlmostEqual(empc, test_val)

    def test_empc_simple_test01(self):
        """Perfect prediction (v1)."""
        y_true = [0, 1]
        y_score = [0, 1]
        empc = EMPChurn(y_true, y_score).empc()
        self.assertAlmostEqual(empc, 28.0000000000391)

    def test_empc_simple_test02(self):
        """Completely incorrect prediction."""
        y_true = [0, 1]
        y_score = [1, 0]
        empc = EMPChurn(y_true, y_score).empc()
        self.assertAlmostEqual(empc, 22.5007912244511)

    def test_empc_simple_test03(self):
        """Half incorrect prediction."""
        y_true = [1, 0]
        y_score = [1, 1]
        empc = EMPChurn(y_true, y_score).empc()
        self.assertAlmostEqual(empc, 22.5007912244511)

    def test_empc_simple_test04(self):
        """Perfect prediction (v2)."""
        y_true = [1, 0]
        y_score = [1, 0]
        empc = EMPChurn(y_true, y_score).empc()
        self.assertAlmostEqual(empc, 28.0000000000391)

    def test_empc_simple_test05(self):
        """Generate QhullError, i.e., ROCCH is equal to diagonal line."""
        y_true = [1, 0]
        y_score = [0.5, 0.5]
        empc = EMPChurn(y_true, y_score).empc()
        self.assertAlmostEqual(empc, 22.5007912244511)

    def test_empc_simple_test06(self):
        """len(np.unique(y_true)) != 2 (all zero)"""
        y_true = [0, 0]
        y_score = [0.25, 0.75]
        with self.assertRaises(AssertionError):
            EMPChurn(y_true, y_score)

    def test_empc_simple_test07(self):
        """len(np.unique(y_true)) != 2 (all one)"""
        y_true = [1, 1]
        y_score = [0.25, 0.75]
        with self.assertRaises(AssertionError):
            EMPChurn(y_true, y_score)

    def test_raise_assertion_error_no_y_score_given(self):
        y = self.y_true
        with self.assertRaises(AssertionError):
            EMPChurn(y_true=y).empc()

    def test_raise_assertion_error_unequal_array_lengths(self):
        y = self.y_true[1:]
        yhat = self.y_score
        with self.assertRaises(AssertionError):
            EMPChurn(y_true=y, y_score=yhat).empc()

    def test_empc_realistic_data(self):
        dat = np.loadtxt("../data/predictions.dat")
        y_true = dat[:, 0]
        y_score = dat[:, 1]
        params = {
            "alpha": 6,
            "beta": 14,
            "clv": 200,
            "d": 10,
            "f": 1,
            "case_label": 1,
        }
        empc = EMPChurn(y_true, y_score, **params).empc()
        self.assertAlmostEqual(empc, 14.9946647690104)

    def test_empc_with_different_parameters_on_realistic_data(self):
        dat = np.loadtxt("../data/predictions.dat")
        y_true = dat[:, 0]
        y_score = dat[:, 1]
        tests = [
            (
                {
                    "alpha": 2,  # changed
                    "beta": 14,
                    "clv": 200,
                    "d": 10,
                    "f": 1,
                },
                5.04518077557418,
            ),
            (
                {
                    "alpha": 6,
                    "beta": 5,  # changed
                    "clv": 200,
                    "d": 10,
                    "f": 1,
                },
                30.2276738879406,
            ),
            (
                {
                    "alpha": 6,
                    "beta": 14,
                    "clv": 50,  # changed
                    "d": 10,
                    "f": 1,
                },
                1.62729137977619,
            ),
            (
                {
                    "alpha": 6,
                    "beta": 14,
                    "clv": 200,
                    "d": 100,  # changed
                    "f": 1,
                },
                1.95497116583216,
            ),
            (
                {
                    "alpha": 6,
                    "beta": 14,
                    "clv": 200,
                    "d": 10,
                    "f": 15,  # changed
                },
                8.47435415117762,
            ),
            (
                {
                    "alpha": 10,  # changed
                    "beta": 7,  # changed
                    "clv": 300,  # changed
                    "d": 50,  # changed
                    "f": 25,  # changed
                },
                23.8505727106226,
            ),
        ]

        for params, test_val in tests:
            empc = EMPChurn(y_true, y_score, **params).empc()
            self.assertAlmostEqual(empc, test_val)


if __name__ == "__main__":
    unittest.main()
