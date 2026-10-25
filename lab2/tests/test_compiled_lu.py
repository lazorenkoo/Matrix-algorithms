import importlib.machinery
import unittest

import numpy as np

try:
    import lab2.lu_compiled as compiled_lu
except ModuleNotFoundError:
    compiled_lu = None


@unittest.skipIf(compiled_lu is None, "Build extensions with python lab2/setup.py build_ext --inplace")
class CompiledLUTests(unittest.TestCase):
    def test_native_dependencies(self):
        import lab2.inverse_compiled as compiled_inverse
        import lab2.multiplication_compiled as compiled_multiplication

        self.assertIsInstance(compiled_lu.__loader__, importlib.machinery.ExtensionFileLoader)
        self.assertIs(compiled_lu._inverse, compiled_inverse._inverse)
        self.assertIs(compiled_lu.multiply, compiled_multiplication.multiply)
        self.assertIs(compiled_lu.get_multiplication, compiled_multiplication.get_multiplication)

    def test_existing_cases(self):
        import lab2.tests.test_lu as cases

        original = cases.lu
        cases.lu = compiled_lu.lu
        try:
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(cases.LUTests)
            result = unittest.TestResult()
            suite.run(result)
            self.assertEqual(result.testsRun, 8)
            self.assertTrue(result.wasSuccessful(), result.errors + result.failures)
        finally:
            cases.lu = original

    def test_same_results_and_counters(self):
        from lab2.lu import lu

        rng = np.random.default_rng(71)
        for size in range(1, 18):
            A = rng.uniform(1e-8, 1, (size, size))
            for method in ("binet", "strassen"):
                L, U, operations = lu(A, method)
                compiled_L, compiled_U, compiled_operations = compiled_lu.lu(A, method)
                np.testing.assert_array_equal(compiled_L, L)
                np.testing.assert_array_equal(compiled_U, U)
                self.assertEqual(compiled_operations, operations)


if __name__ == "__main__":
    unittest.main()
