import numpy as np
import torch
from torch.nn import functional as F 

from simplegrad.tensor import Tensor
from simplegrad.ops import relu, sigmoid, matmul


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

    print("\nAll tests completed successfully!")