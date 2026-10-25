import importlib.machinery
import unittest

import numpy as np

try:
    import lab1.binet_compiled as compiled_binet
    import lab1.strassen_compiled as compiled_strassen
    import lab2.inverse_compiled as compiled_inverse
    import lab2.multiplication_compiled as compiled_multiplication
except ModuleNotFoundError:
    compiled_inverse = None


@unittest.skipIf(compiled_inverse is None, "Build extensions with python lab2/setup.py build_ext --inplace")
class CompiledInverseTests(unittest.TestCase):
    def test_native_dependencies(self):
        for module in (compiled_binet, compiled_strassen, compiled_inverse, compiled_multiplication):
            self.assertIsInstance(module.__loader__, importlib.machinery.ExtensionFileLoader)
        self.assertIs(compiled_inverse.multiply, compiled_multiplication.multiply)
        self.assertIs(compiled_inverse.get_multiplication, compiled_multiplication.get_multiplication)
        import lab2.laderman_compiled as compiled_laderman

        self.assertIs(compiled_multiplication.laderman, compiled_laderman.laderman)
        self.assertIs(compiled_multiplication.binet, compiled_binet.binet)
        self.assertIs(compiled_multiplication.strassen, compiled_strassen.strassen)
        self.assertIs(compiled_multiplication.multiply_direct, compiled_strassen.multiply_direct)

    def test_existing_cases(self):
        import lab2.tests.test_inverse as cases

        original_inverse = cases.inverse
        original_rectangular = cases.strassen_rectangular
        cases.inverse = compiled_inverse.inverse
        cases.strassen_rectangular = compiled_multiplication.strassen_rectangular
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(cases.InverseTests)
            result = unittest.TestResult()
            suite.run(result)
            self.assertEqual(result.testsRun, 8)
            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        finally:
            cases.inverse = original_inverse
            cases.strassen_rectangular = original_rectangular

    def test_same_results_and_counters(self):
        from lab2.inverse import inverse

        rng = np.random.default_rng(91)
        matrices = [rng.uniform(1e-8, 1, (n, n)) for n in range(1, 18)]
        matrices += [np.roll(np.eye(5), 1, axis=0)]
        for A in matrices:
            for method in ("binet", "strassen", "laderman"):
                expected, operations = inverse(A, method)
                actual, compiled_operations = compiled_inverse.inverse(A, method)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(compiled_operations, operations)


if __name__ == "__main__":
    unittest.main()
