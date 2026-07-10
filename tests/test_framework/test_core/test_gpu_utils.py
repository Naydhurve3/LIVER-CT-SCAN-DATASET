import unittest
import numpy as np
import torch
from src.framework.utils.gpu_utils import to_tensor, to_numpy, gpu_report, gpu_clear


class TestToTensor(unittest.TestCase):
    def test_to_tensor_from_numpy(self):
        arr = np.random.randn(3, 64, 64).astype(np.float32)
        tensor = to_tensor(arr)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_to_tensor_from_list(self):
        tensor = to_tensor([1.0, 2.0, 3.0])
        self.assertEqual(tensor.shape, (3,))
        self.assertEqual(tensor.dtype, torch.float32)

    def test_to_tensor_preserves_values(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        tensor = to_tensor(arr)
        self.assertTrue(np.allclose(tensor.numpy(), arr))


class TestToNumpy(unittest.TestCase):
    def test_to_numpy_from_cpu_tensor(self):
        tensor = torch.randn(3, 32, 32)
        arr = to_numpy(tensor)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (3, 32, 32))

    def test_to_numpy_values_preserved(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        arr = to_numpy(tensor)
        self.assertTrue(np.allclose(arr, [[1.0, 2.0], [3.0, 4.0]]))


class TestGPUReport(unittest.TestCase):
    def test_gpu_report_returns_string(self):
        report = gpu_report()
        self.assertIsInstance(report, str)
        self.assertTrue(len(report) > 0)


class TestGPUClear(unittest.TestCase):
    def test_gpu_clear_runs(self):
        try:
            gpu_clear()
        except Exception as e:
            self.fail(f"gpu_clear raised {e}")


if __name__ == "__main__":
    unittest.main()
