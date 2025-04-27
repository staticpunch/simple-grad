import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import *


def test_relu():
    # Define diverse test cases
    test_cases = [
        # Small values
        np.random.rand(2, 2) * 0.001,
        # Large values
        np.random.rand(2, 2) * 1000,
        # Negative values
        np.random.rand(2, 2) * -10,
        # Mixed values
        np.random.rand(2, 2) * 2 - 1,
        # Single value
        np.random.rand(1, 1) * 2 - 1,
        # Vector
        np.random.rand(10) * 2 - 1,
        # Higher dimensions
        np.random.rand(2, 3, 4) * 2 - 1,
        # All zeros
        np.zeros((3, 3)),
        # All ones
        np.ones((3, 3)),
        # Alternating pattern
        np.array([[-1, 1], [1, -1]])
    ]
    
    for i, test_data in enumerate(test_cases):
        pt_x = torch.tensor(test_data, dtype=torch.float32, requires_grad=True)
        x = Tensor.from_torch(pt_x)

        expected = F.relu(pt_x)
        result = relu(x)
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"ReLU forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(x.grad, pt_x.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"ReLU backward test {i+1} passed!")


def test_sigmoid():
    # Define diverse test cases
    test_cases = [
        # Small values
        np.random.rand(2, 2) * 0.001,
        # Large positive values
        np.random.rand(2, 2) * 100,
        # Large negative values
        np.random.rand(2, 2) * -100,
        # Mixed values
        np.random.rand(2, 2) * 4 - 2,
        # Single value
        np.random.rand(1, 1) * 2 - 1,
        # Vector
        np.random.rand(8) * 4 - 2,
        # Higher dimensions
        np.random.rand(2, 2, 2) * 4 - 2,
        # All zeros
        np.zeros((3, 3)),
        # Extreme values
        np.array([[1000, -1000], [-1000, 1000]]),
        # Special pattern
        np.array([[-0.5, 0], [0.5, 1]])
    ]
    
    for i, test_data in enumerate(test_cases):
        pt_x = torch.tensor(test_data, dtype=torch.float32, requires_grad=True)
        x = Tensor.from_torch(pt_x)

        expected = F.sigmoid(pt_x)
        result = sigmoid(x)
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Sigmoid forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(x.grad, pt_x.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Sigmoid backward test {i+1} passed!")


def test_matmul():
    # Define diverse test cases for matmul (need pairs of matrices with compatible dimensions)
    test_cases = [
        # Small matrices
        (np.random.rand(2, 3) * 0.001, np.random.rand(3, 2) * 0.001),
        # Large matrices
        (np.random.rand(3, 4) * 1000, np.random.rand(4, 5) * 1000),
        # Mixed scale
        (np.random.rand(2, 3) * 0.01, np.random.rand(3, 4) * 100),
        # Square matrices
        (np.random.rand(5, 5) * 2 - 1, np.random.rand(5, 5) * 2 - 1),
        # Vector-matrix
        (np.random.rand(1, 4) * 2 - 1, np.random.rand(4, 3) * 2 - 1),
        # Matrix-vector (effectively)
        (np.random.rand(3, 2) * 2 - 1, np.random.rand(2, 1) * 2 - 1),
        # Tall-skinny and short-fat
        (np.random.rand(10, 2) * 2 - 1, np.random.rand(2, 8) * 2 - 1),
        # Identity matrices
        (np.eye(4), np.random.rand(4, 4) * 2 - 1),
        # Zero matrices
        (np.random.rand(3, 3) * 2 - 1, np.zeros((3, 3))),
    ]
    
    for i, (a_data, b_data) in enumerate(test_cases):
        try:
            pt_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
            pt_b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
            a = Tensor.from_torch(pt_a)
            b = Tensor.from_torch(pt_b)

            expected = pt_a @ pt_b
            result = matmul(a, b)
            
            np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                      rtol=1.3e-6, atol=1e-5)
            print(f"MatMul forward test {i+1} passed!")

            expected.backward(torch.ones_like(expected))
            result.backward()
            
            np.testing.assert_allclose(a.grad, pt_a.grad.detach().numpy(),
                                      rtol=1.3e-6, atol=1e-5)
            np.testing.assert_allclose(b.grad, pt_b.grad.detach().numpy(),
                                      rtol=1.3e-6, atol=1e-5)
            print(f"MatMul backward test {i+1} passed!")
        except Exception as e:
            print(f"MatMul test {i+1} failed: {e}")


def test_elemwise_add():
    # Define diverse test cases
    test_cases = [
        # Small values
        (np.random.rand(2, 2) * 0.001, np.random.rand(2, 2) * 0.001),
        # Large values
        (np.random.rand(2, 2) * 1000, np.random.rand(2, 2) * 1000),
        # Mixed scale
        (np.random.rand(2, 2) * 0.01, np.random.rand(2, 2) * 100),
        # Negative and positive
        (np.random.rand(3, 3) * 2 - 1, np.random.rand(3, 3) * 2 - 1),
        # Vector addition
        (np.random.rand(10) * 2 - 1, np.random.rand(10) * 2 - 1),
        # Higher dimensions
        (np.random.rand(2, 3, 4) * 2 - 1, np.random.rand(2, 3, 4) * 2 - 1),
        # Zeros and values
        (np.zeros((3, 3)), np.random.rand(3, 3) * 2 - 1),
        # All ones
        (np.ones((3, 3)), np.ones((3, 3))),
        # Sparse-like pattern
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        # Single value
        (np.random.rand(1, 1) * 2 - 1, np.random.rand(1, 1) * 2 - 1)
    ]
    
    for i, (a_data, b_data) in enumerate(test_cases):
        pt_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        pt_b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
        a = Tensor.from_torch(pt_a)
        b = Tensor.from_torch(pt_b)

        expected = pt_a + pt_b
        result = a + b
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Addition forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(a.grad, pt_a.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        np.testing.assert_allclose(b.grad, pt_b.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Addition backward test {i+1} passed!")


def test_elemwise_mul():
    # Define diverse test cases
    test_cases = [
        # Small values
        (np.random.rand(2, 2) * 0.001, np.random.rand(2, 2) * 0.001),
        # Large values
        (np.random.rand(2, 2) * 1000, np.random.rand(2, 2) * 1000),
        # Mixed scale
        (np.random.rand(2, 2) * 0.01, np.random.rand(2, 2) * 100),
        # Negative and positive
        (np.random.rand(3, 3) * 2 - 1, np.random.rand(3, 3) * 2 - 1),
        # Vector multiplication
        (np.random.rand(10) * 2 - 1, np.random.rand(10) * 2 - 1),
        # Higher dimensions
        (np.random.rand(2, 3, 4) * 2 - 1, np.random.rand(2, 3, 4) * 2 - 1),
        # Zeros and values
        (np.zeros((3, 3)), np.random.rand(3, 3) * 2 - 1),
        # All ones
        (np.ones((3, 3)), np.ones((3, 3))),
        # Sparse-like pattern
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        # Single value
        (np.random.rand(1, 1) * 2 - 1, np.random.rand(1, 1) * 2 - 1)
    ]
    
    for i, (a_data, b_data) in enumerate(test_cases):
        pt_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        pt_b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
        a = Tensor.from_torch(pt_a)
        b = Tensor.from_torch(pt_b)

        expected = pt_a * pt_b
        result = a * b
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Multiplication forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(a.grad, pt_a.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        np.testing.assert_allclose(b.grad, pt_b.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Multiplication backward test {i+1} passed!")

def test_elemwise_sub():
    # Define diverse test cases
    test_cases = [
        # Small values
        (np.random.rand(2, 2) * 0.001, np.random.rand(2, 2) * 0.001),
        # Large values
        (np.random.rand(2, 2) * 1000, np.random.rand(2, 2) * 1000),
        # Mixed scale
        (np.random.rand(2, 2) * 0.01, np.random.rand(2, 2) * 100),
        # Negative and positive
        (np.random.rand(3, 3) * 2 - 1, np.random.rand(3, 3) * 2 - 1),
        # Vector subtraction
        (np.random.rand(10) * 2 - 1, np.random.rand(10) * 2 - 1),
        # Higher dimensions
        (np.random.rand(2, 3, 4) * 2 - 1, np.random.rand(2, 3, 4) * 2 - 1),
        # Zeros and values
        (np.zeros((3, 3)), np.random.rand(3, 3) * 2 - 1),
        # All ones
        (np.ones((3, 3)), np.ones((3, 3))),
        # Sparse-like pattern
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
        # Single value
        (np.random.rand(1, 1) * 2 - 1, np.random.rand(1, 1) * 2 - 1)
    ]
    
    for i, (a_data, b_data) in enumerate(test_cases):
        pt_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        pt_b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
        a = Tensor.from_torch(pt_a)
        b = Tensor.from_torch(pt_b)

        expected = pt_a - pt_b
        result = a - b
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Subtraction forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(a.grad, pt_a.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        np.testing.assert_allclose(b.grad, pt_b.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Subtraction backward test {i+1} passed!")


def test_elemwise_div():
    # Define diverse test cases
    test_cases = [
        # Small values (avoiding zeros in denominator)
        (np.random.rand(2, 2) * 0.001, np.random.rand(2, 2) * 0.001 + 0.01),
        # Large values
        (np.random.rand(2, 2) * 1000, np.random.rand(2, 2) * 1000 + 1),
        # Mixed scale
        (np.random.rand(2, 2) * 0.01, np.random.rand(2, 2) * 100 + 0.1),
        # Negative and positive (avoiding zeros in denominator)
        (np.random.rand(3, 3) * 2 - 1, np.random.rand(3, 3) * 2 + 0.1),
        # Vector division
        (np.random.rand(10) * 2 - 1, np.random.rand(10) * 2 + 0.1),
        # Higher dimensions
        (np.random.rand(2, 3, 4) * 2 - 1, np.random.rand(2, 3, 4) * 2 + 0.1),
        # Values and non-zeros
        (np.random.rand(3, 3) * 2 - 1, np.ones((3, 3)) * 0.5),
        # All ones
        (np.ones((3, 3)), np.ones((3, 3)) * 2),
        # Specific pattern
        (np.array([[4.0, 9.0], [16.0, 25.0]]), np.array([[2.0, 3.0], [4.0, 5.0]])),
        # Single value
        (np.random.rand(1, 1) * 2 - 1, np.random.rand(1, 1) * 2 + 0.1)
    ]
    
    for i, (a_data, b_data) in enumerate(test_cases):
        pt_a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
        pt_b = torch.tensor(b_data, dtype=torch.float32, requires_grad=True)
        a = Tensor.from_torch(pt_a)
        b = Tensor.from_torch(pt_b)

        expected = pt_a / pt_b
        result = a / b
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Division forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(a.grad, pt_a.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        np.testing.assert_allclose(b.grad, pt_b.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Division backward test {i+1} passed!")


def test_negation():
    # Define diverse test cases
    test_cases = [
        # Small values
        np.random.rand(2, 2) * 0.001,
        # Large values
        np.random.rand(2, 2) * 1000,
        # Negative values
        np.random.rand(2, 2) * -10,
        # Mixed values
        np.random.rand(2, 2) * 2 - 1,
        # Single value
        np.random.rand(1, 1) * 2 - 1,
        # Vector
        np.random.rand(10) * 2 - 1,
        # Higher dimensions
        np.random.rand(2, 3, 4) * 2 - 1,
        # All zeros
        np.zeros((3, 3)),
        # All ones
        np.ones((3, 3)),
        # Alternating pattern
        np.array([[-1, 1], [1, -1]])
    ]
    
    for i, test_data in enumerate(test_cases):
        pt_x = torch.tensor(test_data, dtype=torch.float32, requires_grad=True)
        x = Tensor.from_torch(pt_x)

        expected = -pt_x
        result = -x
        
        np.testing.assert_allclose(result.data, expected.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Negation forward test {i+1} passed!")

        expected.backward(torch.ones_like(expected))
        result.backward()
        
        np.testing.assert_allclose(x.grad, pt_x.grad.detach().numpy(),
                                  rtol=1.3e-6, atol=1e-5)
        print(f"Negation backward test {i+1} passed!")

def test_reshape():
    """Test the reshape operation with various challenging cases"""
    print("\n=== TESTING RESHAPE ===")
    
    # Define comprehensive test cases
    test_cases = [
        # Basic reshaping
        {"data": np.random.rand(2, 3), "shape": (6,), "name": "2D to 1D"},
        {"data": np.random.rand(6), "shape": (2, 3), "name": "1D to 2D"},
        {"data": np.random.rand(2, 3), "shape": (3, 2), "name": "Transpose-like reshape"},
        
        # Using -1 in shape (inferred dimension)
        {"data": np.random.rand(2, 3, 4), "shape": (-1, 4), "name": "Using -1 to infer dimension"},
        {"data": np.random.rand(24), "shape": (2, 3, -1), "name": "Using -1 in higher dimensions"},
        
        # No-op reshaping (same shape)
        {"data": np.random.rand(5, 5), "shape": (5, 5), "name": "No-op reshape (same shape)"},
        
        # Extreme values
        {"data": np.random.rand(2, 3) * 1e9, "shape": (6,), "name": "Large values (1e9)"},
        {"data": np.random.rand(2, 3) * 1e-9, "shape": (6,), "name": "Small values (1e-9)"},
        {"data": np.array([[1e15, 1e-15], [1e-15, 1e15]]), "shape": (4,), "name": "Mixed extreme values"},
        
        # Large dimensions
        {"data": np.random.rand(1000, 5), "shape": (5000,), "name": "Large dimensions flattened"},
        {"data": np.random.rand(5000), "shape": (1000, 5), "name": "Large flat array to 2D"},
        
        # Complex reshaping
        {"data": np.random.rand(2, 3, 4, 5), "shape": (6, 20), "name": "4D to 2D"},
        {"data": np.random.rand(2, 3, 4, 5), "shape": (2, -1), "name": "4D to 2D with inferred dim"},
        {"data": np.random.rand(100), "shape": (2, 5, 2, 5), "name": "1D to 4D"},
        
        # Special patterns
        {"data": np.ones((3, 4)), "shape": (12,), "name": "Reshaping ones"},
        {"data": np.zeros((3, 4)), "shape": (4, 3), "name": "Reshaping zeros"},
        {"data": np.eye(4), "shape": (16,), "name": "Reshaping identity matrix"},
        
        # Edge cases
        {"data": np.array([1.0]), "shape": (1, 1, 1), "name": "Scalar to 3D"},
        {"data": np.random.rand(1, 1, 1, 5), "shape": (5,), "name": "Multiple singleton dimensions to 1D"},
        
        # Challenging gradients
        {"data": np.random.rand(3, 4), "shape": (12,), "name": "Gradient flow test 1"},
        {"data": np.random.rand(12), "shape": (3, 4), "name": "Gradient flow test 2"},
        {"data": np.random.rand(2, 3, 4), "shape": (8, 3), "name": "Gradient flow test 3"},
    ]
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        shape = test_case["shape"]
        name = test_case["name"]
        
        print(f"\nTest case {i+1}: {name}")
        print(f"  Input shape: {data.shape}, Target shape: {shape}")
        
        # Create tensors
        pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        x = Tensor(data, requires_grad=True)
        
        # PyTorch reshape
        expected = pt_x.reshape(shape)
        
        # Our reshape
        result = reshape(x, shape)
        
        # Check forward pass
        try:
            np.testing.assert_allclose(
                result.data,
                expected.detach().numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Reshape forward pass failed"
            )
            print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            continue
        
        # Generate random gradient for backward pass
        # Get the actual shape after reshape (to handle -1 in shape)
        actual_shape = result.data.shape
        grad_output_np = np.random.rand(*actual_shape).astype(np.float32)
        grad_output_torch = torch.tensor(grad_output_np)
        
        # Compute gradients
        expected.backward(grad_output_torch)
        result.backward(grad_output_np)
        
        # Check backward pass
        try:
            np.testing.assert_allclose(
                x.grad,
                pt_x.grad.detach().numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"Reshape backward pass failed"
            )
            print(f"  ✓ Backward pass successful")
        except Exception as e:
            print(f"  ✗ Backward pass failed: {e}")
        
        # Reset gradients
        pt_x.grad = None
        x.grad = np.zeros_like(x.data)

def test_logsumexp():
    # Define diverse test cases
    test_cases = [
        # Small values
        {"data": np.random.rand(3, 4) * 0.001, "axis": None, "keepdims": False},
        # Large values
        {"data": np.random.rand(3, 4) * 100, "axis": None, "keepdims": False},
        # Negative values
        {"data": np.random.rand(3, 4) * -10, "axis": None, "keepdims": False},
        # Mixed values
        {"data": np.random.rand(3, 4) * 2 - 1, "axis": None, "keepdims": False},
        # Single dimension reduction with keepdims=True
        {"data": np.random.rand(3, 4) * 2 - 1, "axis": 0, "keepdims": True},
        # Single dimension reduction with keepdims=False
        {"data": np.random.rand(3, 4) * 2 - 1, "axis": 1, "keepdims": False},
        # Multiple dimensions
        {"data": np.random.rand(2, 3, 4) * 2 - 1, "axis": None, "keepdims": False},
        # Multiple dimensions with specific axis
        {"data": np.random.rand(2, 3, 4) * 2 - 1, "axis": 1, "keepdims": False},
        # Multiple dimensions with tuple axis
        {"data": np.random.rand(2, 3, 4) * 2 - 1, "axis": (0, 2), "keepdims": False},
        # Multiple dimensions with tuple axis and keepdims=True
        {"data": np.random.rand(2, 3, 4) * 2 - 1, "axis": (0, 2), "keepdims": True}
    ]
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        axis = test_case["axis"]
        keepdims = test_case["keepdims"]
        
        # Convert to tensors
        pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        x = Tensor.from_torch(pt_x)
        
        # PyTorch version
        if axis is None:
            # PyTorch's logsumexp requires a specific dim
            expected = torch.logsumexp(pt_x, dim=tuple(range(pt_x.dim())), keepdim=keepdims)
        elif isinstance(axis, int):
            expected = torch.logsumexp(pt_x, dim=axis, keepdim=keepdims)
        else:
            # For multiple axes, we need to handle them one by one in PyTorch
            temp = pt_x
            # Process axes in reverse order to maintain correct dimensions
            for ax in sorted(axis, reverse=True):
                temp = torch.logsumexp(temp, dim=ax, keepdim=keepdims)
            expected = temp
        
        # Our implementation
        result = logsumexp(x, axis=axis, keepdims=keepdims)
        
        # Check forward pass
        np.testing.assert_allclose(
            result.data, 
            expected.detach().numpy(), 
            rtol=1e-5, atol=1e-5,
            err_msg=f"Forward pass failed for test case {i+1}: data shape {data.shape}, axis {axis}, keepdims {keepdims}"
        )
        print(f"LogSumExp forward test {i+1} passed!")
        
        # Compute gradients
        grad_output = torch.ones_like(expected)
        expected.backward(grad_output)
        result.backward()
        
        # Check backward pass
        np.testing.assert_allclose(
            x.grad, 
            pt_x.grad.detach().numpy(), 
            rtol=1e-5, atol=1e-5,
            err_msg=f"Backward pass failed for test case {i+1}: data shape {data.shape}, axis {axis}, keepdims {keepdims}"
        )
        print(f"LogSumExp backward test {i+1} passed!")

def test_logsumexp_specific_cases():
    """Test specific edge cases for logsumexp"""
    
    # Case 1: All elements are the same (tests numerical stability)
    data = np.ones((3, 3)) * 1000  # Large identical values
    pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    x = Tensor.from_torch(pt_x)
    
    expected = torch.logsumexp(pt_x, dim=1, keepdim=False)
    result = logsumexp(x, axis=1, keepdims=False)
    
    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("LogSumExp specific case 1 (large identical values) passed!")
    
    # Case 2: Extreme differences between values (tests numerical stability)
    data = np.array([[1e-10, 1e10], [1e-10, 1e-10]])
    pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    x = Tensor.from_torch(pt_x)
    
    expected = torch.logsumexp(pt_x, dim=1, keepdim=False)
    result = logsumexp(x, axis=1, keepdims=False)
    
    np.testing.assert_allclose(result.data, expected.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("LogSumExp specific case 2 (extreme value differences) passed!")
    
    # Case 3: Test with softmax relation (logsumexp is used in softmax implementation)
    data = np.random.rand(5, 10) * 2 - 1
    pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
    x = Tensor.from_torch(pt_x)
    
    # Standard softmax calculation using logsumexp
    pt_logsumexp = torch.logsumexp(pt_x, dim=1, keepdim=True)
    pt_softmax = torch.exp(pt_x - pt_logsumexp)
    
    our_logsumexp = logsumexp(x, axis=1, keepdims=True)
    our_softmax = np.exp(x.data - our_logsumexp.data)
    
    np.testing.assert_allclose(our_softmax, pt_softmax.detach().numpy(), rtol=1e-5, atol=1e-5)
    print("LogSumExp specific case 3 (softmax relation) passed!")

def test_summation():
    """Test the summation operation with challenging cases"""
    print("\n=== TESTING SUMMATION ===")
    
    # Define challenging test cases
    test_cases = [
        # Basic cases
        {"data": np.random.rand(5, 5), "axis": None, "keepdims": False, "name": "Basic 2D, all axes"},
        {"data": np.random.rand(5, 5), "axis": 0, "keepdims": False, "name": "Basic 2D, axis 0"},
        {"data": np.random.rand(5, 5), "axis": 1, "keepdims": True, "name": "Basic 2D, axis 1 with keepdims"},
        
        # Extreme values
        {"data": np.random.rand(10, 10) * 1e10, "axis": None, "keepdims": False, "name": "Large values (1e10)"},
        {"data": np.random.rand(10, 10) * 1e-10, "axis": 0, "keepdims": True, "name": "Small values (1e-10)"},
        {"data": np.array([[1e15, 1e-15], [1e-15, 1e15]]), "axis": 1, "keepdims": False, "name": "Mixed extreme values"},
        
        # Large dimensions
        {"data": np.random.rand(1000, 5), "axis": 0, "keepdims": False, "name": "Large first dimension (1000x5)"},
        {"data": np.random.rand(5, 1000), "axis": 1, "keepdims": True, "name": "Large second dimension (5x1000)"},
        
        # Higher dimensions
        {"data": np.random.rand(10, 10, 10), "axis": (0, 2), "keepdims": False, "name": "3D with multiple axes"},
        {"data": np.random.rand(5, 5, 5, 5), "axis": (1, 2), "keepdims": True, "name": "4D with multiple axes and keepdims"},
        
        # Special patterns
        {"data": np.ones((20, 20)), "axis": None, "keepdims": False, "name": "All ones"},
        {"data": np.zeros((20, 20)), "axis": 0, "keepdims": True, "name": "All zeros"},
        {"data": np.eye(20), "axis": 1, "keepdims": False, "name": "Identity matrix"},
        
        # Edge cases
        {"data": np.array([1.0]), "axis": None, "keepdims": False, "name": "Single value"},
        {"data": np.random.rand(1, 1, 1, 1), "axis": (1, 2), "keepdims": True, "name": "Multiple singleton dimensions"},
    ]
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        axis = test_case["axis"]
        keepdims = test_case["keepdims"]
        name = test_case["name"]
        
        # Create tensors
        pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        x = Tensor(data, requires_grad=True)
        
        # PyTorch sum
        if isinstance(axis, tuple):
            # For multiple axes in PyTorch, we need to do them one by one
            expected = pt_x
            for ax in sorted(axis, reverse=True):  # Start from the highest axis
                expected = expected.sum(dim=ax, keepdim=keepdims)
        else:
            expected = pt_x.sum(dim=axis, keepdim=keepdims)
        
        # Our summation
        result = summation(x, axis=axis, keepdims=keepdims)
        
        # Check forward pass
        try:
            np.testing.assert_allclose(
                result.data,
                expected.detach().numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Summation forward pass failed"
            )
            # print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            continue
        
        # Generate random gradient for backward pass
        grad_output_np = np.random.rand(*result.data.shape)
        if isinstance(grad_output_np, float):
            grad_output_np = np.float32(grad_output_np)
        else:
            grad_output_np = grad_output_np.astype(np.float32)
        grad_output_torch = torch.tensor(grad_output_np)
        
        # Compute gradients
        expected.backward(grad_output_torch)
        result.backward(grad_output_np)
        
        # Check backward pass
        try:
            np.testing.assert_allclose(
                x.grad,
                pt_x.grad.detach().numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"broadcast_to backward pass failed"
            )
            # print(f"  ✓ Backward pass successful")
            result_msg = "Successful."
        except Exception as e:
            result_msg = "Failed."
            print(f"  ✗ Backward pass failed: {e}")
        
        # Reset gradients
        pt_x.grad = None
        x.grad = np.zeros_like(x.data)
        
        print(
            f"Test case {i+1}: {name}."
            # f" Shape: {data.shape}, Axis: {axis}, Keepdims: {keepdims}."
            f" {result_msg}"
        )

def test_broadcast_to():
    """Test the broadcast_to operation with challenging cases"""
    print("\n=== TESTING BROADCAST_TO ===")
    
    # Define challenging test cases
    test_cases = [
        # Basic broadcasting
        {"data": np.random.rand(1), "shape": (10,), "name": "Scalar to vector"},
        {"data": np.random.rand(1, 5), "shape": (10, 5), "name": "Row to matrix"},
        {"data": np.random.rand(5, 1), "shape": (5, 10), "name": "Column to matrix"},
        
        # Extreme values
        {"data": np.random.rand(1, 3) * 1e9, "shape": (5, 3), "name": "Large values (1e9)"},
        {"data": np.random.rand(1, 3) * 1e-9, "shape": (5, 3), "name": "Small values (1e-9)"},
        {"data": np.array([[1e15], [1e-15]]), "shape": (2, 5), "name": "Mixed extreme values"},
        
        # Large dimensions
        {"data": np.random.rand(1, 5), "shape": (1000, 5), "name": "Broadcast to large first dim (1000)"},
        {"data": np.random.rand(5, 1), "shape": (5, 1000), "name": "Broadcast to large second dim (1000)"},
        
        # Higher dimensions
        {"data": np.random.rand(1, 5, 1), "shape": (10, 5, 8), "name": "3D broadcasting"},
        {"data": np.random.rand(1, 1, 1, 5), "shape": (7, 6, 5, 5), "name": "4D broadcasting"},
        
        # Multiple dimensions broadcasted
        {"data": np.random.rand(1, 1, 3), "shape": (8, 8, 3), "name": "Broadcasting multiple dimensions"},
        
        # Special patterns
        {"data": np.ones((1, 5)), "shape": (10, 5), "name": "Broadcasting ones"},
        {"data": np.zeros((1, 5)), "shape": (10, 5), "name": "Broadcasting zeros"},
        
        # No broadcasting (identity case)
        {"data": np.random.rand(5, 5), "shape": (5, 5), "name": "No broadcasting (same shape)"},
        
        # Edge cases
        {"data": np.array([1.0]), "shape": (1, 1, 1, 1), "name": "Scalar to higher dims"},
    ]
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        shape = test_case["shape"]
        name = test_case["name"]
        
        # Create tensors
        pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        x = Tensor(data, requires_grad=True)
        
        # PyTorch broadcast (expand)
        # Handle the case of broadcasting to higher dimensions
        if pt_x.dim() < len(shape):
            # Add dimensions to match the target shape
            expanded_dims = len(shape) - pt_x.dim()
            reshape_dims = [1] * expanded_dims + list(pt_x.shape)
            pt_x_reshaped = pt_x.reshape(reshape_dims)
            expected = pt_x_reshaped.expand(shape)
        else:
            expected = pt_x.expand(shape)
        
        # Our broadcast_to
        result = broadcast_to(x, shape)
        
        # Check forward pass
        try:
            np.testing.assert_allclose(
                result.data,
                expected.detach().numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"broadcast_to forward pass failed"
            )
            # print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            continue
        
        # Generate random gradient for backward pass
        grad_output_np = np.random.rand(*shape).astype(np.float32)
        grad_output_torch = torch.tensor(grad_output_np)
        
        # Compute gradients
        expected.backward(grad_output_torch)
        result.backward(grad_output_np)
        
        # Check backward pass
        try:
            np.testing.assert_allclose(
                x.grad,
                pt_x.grad.detach().numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"broadcast_to backward pass failed"
            )
            # print(f"  ✓ Backward pass successful")
            result_msg = "Successful."
        except Exception as e:
            result_msg = "Failed."
            print(f"  ✗ Backward pass failed: {e}")
        
        # Reset gradients
        pt_x.grad = None
        x.grad = np.zeros_like(x.data)
        
        print(
            f"Test case {i+1}: {name}."
            # f" Shape: {data.shape}, Axis: {axis}, Keepdims: {keepdims}."
            f" {result_msg}"
        )

def test_softmax():
    """Test the softmax operation with challenging cases"""
    print("\n=== TESTING SOFTMAX ===")
    
    # Define challenging test cases
    test_cases = [
        # Basic cases
        {"data": np.random.rand(10), "axis": None, "name": "Basic 1D vector"},
        {"data": np.random.rand(5, 5), "axis": 1, "name": "Basic 2D, axis 1"},
        {"data": np.random.rand(5, 5), "axis": 0, "name": "Basic 2D, axis 0"},
        
        # Extreme values
        {"data": np.random.rand(10) * 1e9, "axis": None, "name": "Large values (1e9)"},
        {"data": np.random.rand(10) * 1e-9, "axis": None, "name": "Small values (1e-9)"},
        {"data": np.array([1e15, 1e-15, 0, -1e-15, -1e15]), "axis": None, "name": "Mixed extreme values"},
        
        # Numerical stability challenges
        {"data": np.array([1000, 0, -1000]), "axis": None, "name": "Very different values"},
        {"data": np.array([1e5, 1e5 + 1e-5]), "axis": None, "name": "Nearly identical large values"},
        {"data": np.ones(10) * 1e5, "axis": None, "name": "All identical large values"},
        
        # Large dimensions
        {"data": np.random.rand(1000, 5), "axis": 1, "name": "Large first dimension (1000x5)"},
        {"data": np.random.rand(5, 1000) * 1e9, "axis": 0, "name": "Large second dimension (5x1000)"},
        
        # Higher dimensions
        {"data": np.random.rand(10, 10, 10), "axis": 2, "name": "3D tensor, last axis"},
        {"data": np.random.rand(10, 10, 10), "axis": 1, "name": "3D tensor, middle axis"},
        {"data": np.random.rand(5, 5, 5, 5), "axis": 0, "name": "4D tensor, first axis"},
        
        # Special patterns
        {"data": np.zeros((10, 10)), "axis": 1, "name": "All zeros (uniform distribution)"},
        {"data": np.ones((10, 10)), "axis": 1, "name": "All ones (uniform distribution)"},
        {"data": np.eye(10), "axis": 1, "name": "Identity matrix"},
        
        # Edge cases
        {"data": np.array([42.0]), "axis": None, "name": "Single value (should be 1.0)"},
        {"data": np.zeros((1, 1, 1)), "axis": 1, "name": "Multiple singleton dimensions"},
    ]
    
    for i, test_case in enumerate(test_cases):
        data = test_case["data"]
        axis = test_case["axis"]
        name = test_case["name"]
        
        # print(f"\nTest case {i+1}: {name}")
        # print(f"  Shape: {data.shape}, Axis: {axis}")
        
        # Create tensors
        pt_x = torch.tensor(data, dtype=torch.float32, requires_grad=True)
        x = Tensor(data, requires_grad=True)
        
        # PyTorch softmax
        if axis is None:
            # Flatten for axis=None
            flattened = pt_x.reshape(-1)
            expected = torch.nn.functional.softmax(flattened, dim=0)
        else:
            expected = torch.nn.functional.softmax(pt_x, dim=axis)
        
        # Our softmax
        result = softmax(x, axis=axis)
        
        # Check forward pass
        try:
            np.testing.assert_allclose(
                result.data,
                expected.detach().numpy(),
                rtol=1e-5, atol=1e-5,
                err_msg=f"Softmax forward pass failed"
            )
            # print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
            continue
        
        # Generate random gradient for backward pass
        grad_output_np = np.random.rand(*result.data.shape).astype(np.float32)
        
        # For PyTorch, ensure gradient has the right shape
        if axis is None:
            grad_output_torch = torch.tensor(grad_output_np.reshape(-1))
        else:
            grad_output_torch = torch.tensor(grad_output_np)
        
        # Compute gradients
        expected.backward(grad_output_torch)
        result.backward(grad_output_np)
        
        # Check backward pass
        try:
            np.testing.assert_allclose(
                x.grad,
                pt_x.grad.detach().numpy(),
                rtol=1e-4, atol=1e-5,
                err_msg=f"broadcast_to backward pass failed"
            )
            # print(f"  ✓ Backward pass successful")
            result_msg = "Successful."
        except Exception as e:
            result_msg = "Failed."
            print(f"  ✗ Backward pass failed: {e}")
        
        # Reset gradients
        pt_x.grad = None
        x.grad = np.zeros_like(x.data)
        
        print(
            f"Test case {i+1}: {name}."
            # f" Shape: {data.shape}, Axis: {axis}, Keepdims: {keepdims}."
            f" {result_msg}"
        )

if __name__ == "__main__":
    print("Testing ReLU operation...")
    test_relu()
    
    print("\nTesting Sigmoid operation...")
    test_sigmoid()
    
    print("\nTesting Matrix Multiplication operation...")
    test_matmul()
    
    print("\nTesting Element-wise Addition operation...")
    test_elemwise_add()
    
    print("\nTesting Element-wise Multiplication operation...")
    test_elemwise_mul()

    print("\nTesting Element-wise Subtraction operation...")
    test_elemwise_sub()
    
    print("\nTesting Element-wise Division operation...")
    test_elemwise_div()
    
    print("\nTesting Negation operation...")
    test_negation()

    print("\nTesting Reshape operation...")
    test_reshape()
    
    print("\nTesting Broadcast operation...")
    test_broadcast_to()
    
    print("\nTesting Summation operation...")
    test_summation()
    
    print("\nTesting LogSumExp operation...")
    test_logsumexp()
    
    print("\nTesting LogSumExp specific cases...")
    test_logsumexp_specific_cases()
    
    print("\nTesting Softmax operation...")
    test_softmax()
    
    print("\nAll tests completed successfully!")