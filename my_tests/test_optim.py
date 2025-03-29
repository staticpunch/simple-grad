import torch
import torch.nn as nn
import simplegrad as sg
import simplegrad.module as snn
from simplegrad.optim import NaiveSGD, Adam
from simplegrad.ops import *

def test_linear_fwd_bwd():
    # Test parameters
    in_features, out_features = 30, 40
    batch_size = 2
    lr = 0.005
    
    # Pre-generate weights for identical initialization
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
    bias = np.zeros(out_features, dtype=np.float32)
    
    # Initialize your implementation with fixed weights
    linear = snn.Linear(in_features, out_features)
    linear.weight.data = weight.copy()
    linear.bias.data = bias.copy()
    
    # Initialize PyTorch equivalent with same weights
    torch_linear = nn.Linear(in_features, out_features, bias=True)
    torch_linear.weight.data = torch.tensor(weight, dtype=torch.float32)
    torch_linear.bias.data = torch.tensor(bias, dtype=torch.float32)
    
    # Generate random input
    np_x = np.random.randn(batch_size, in_features).astype(np.float32)
    x = snn.Tensor(np_x, requires_grad=False)
    torch_x = torch.tensor(np_x, requires_grad=False, dtype=torch.float32)
    
    # Random target
    np_y = np.random.randn(batch_size, out_features).astype(np.float32)
    y = snn.Tensor(np_y, requires_grad=False)
    torch_y = torch.tensor(np_y, requires_grad=False, dtype=torch.float32)
    
    # Forward pass
    output = linear(x)
    torch_output = torch_linear(torch_x)
    # return output, torch_output
    # Check output
    np.testing.assert_allclose(output.data, torch_output.detach().numpy(), rtol=1e-4)
    print("Forward pass matches PyTorch ✓")
    
    # Loss
    loss = summation((output - y) ** 2)
    torch_loss = ((torch_output - torch_y) ** 2).sum()
    
    # Backward pass
    loss.backward()
    torch_loss.backward()
    
    # Check gradients
    np.testing.assert_allclose(linear.weight.grad, torch_linear.weight.grad.numpy(), rtol=1e-4)
    np.testing.assert_allclose(linear.bias.grad, torch_linear.bias.grad.numpy(), rtol=1e-4)
    print("Gradients match PyTorch ✓")


def test_sgd():
    # Test parameters
    in_features, out_features = 30, 40
    batch_size = 2
    lr = 0.005
    
    # Pre-generate weights for identical initialization
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
    bias = np.zeros(out_features, dtype=np.float32)
    
    # Initialize your implementation with fixed weights
    linear = snn.Linear(in_features, out_features)
    linear.weight.data = weight.copy()
    linear.bias.data = bias.copy()
    
    # Initialize PyTorch equivalent with same weights
    torch_linear = nn.Linear(in_features, out_features, bias=True)
    torch_linear.weight.data = torch.tensor(weight, dtype=torch.float32)
    torch_linear.bias.data = torch.tensor(bias, dtype=torch.float32)
    
    # Generate random input
    np_x = np.random.randn(batch_size, in_features).astype(np.float32)
    x = snn.Tensor(np_x, requires_grad=False)
    torch_x = torch.tensor(np_x, requires_grad=False, dtype=torch.float32)
    
    # Random target
    np_y = np.random.randn(batch_size, out_features).astype(np.float32)
    y = snn.Tensor(np_y, requires_grad=False)
    torch_y = torch.tensor(np_y, requires_grad=False, dtype=torch.float32)
    
    # Forward pass
    output = linear(x)
    torch_output = torch_linear(torch_x)
    
    # Loss
    loss = summation((output - y) ** 2)
    torch_loss = ((torch_output - torch_y) ** 2).sum()
    
    # Backward pass
    loss.backward()
    torch_loss.backward()
    
    # Optimizer step
    opt = NaiveSGD([linear.weight, linear.bias], lr=lr)
    torch_opt = torch.optim.SGD(torch_linear.parameters(), lr=lr)
    
    opt.step()
    torch_opt.step()
    
    # Check updated weights
    np.testing.assert_allclose(linear.weight.data, torch_linear.weight.data.numpy(), rtol=1e-4)
    np.testing.assert_allclose(linear.bias.data, torch_linear.bias.data.numpy(), rtol=1e-4)
    print("SGD: Parameter updates match PyTorch ✓")
    
    print("SGD test passed!")


def test_adam():
    # Test parameters
    in_features, out_features = 30, 40
    batch_size = 2
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-8
    
    # Pre-generate weights for identical initialization
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.01
    bias = np.zeros(out_features, dtype=np.float32)
    
    # Initialize your implementation with fixed weights
    linear = snn.Linear(in_features, out_features)
    linear.weight.data = weight.copy()
    linear.bias.data = bias.copy()
    
    # Initialize PyTorch equivalent with same weights
    torch_linear = nn.Linear(in_features, out_features, bias=True)
    torch_linear.weight.data = torch.tensor(weight, dtype=torch.float32)
    torch_linear.bias.data = torch.tensor(bias, dtype=torch.float32)
    
    # Generate random input
    np_x = np.random.randn(batch_size, in_features).astype(np.float32)
    x = snn.Tensor(np_x, requires_grad=False)
    torch_x = torch.tensor(np_x, requires_grad=False, dtype=torch.float32)
    
    # Random target
    np_y = np.random.randn(batch_size, out_features).astype(np.float32)
    y = snn.Tensor(np_y, requires_grad=False)
    torch_y = torch.tensor(np_y, requires_grad=False, dtype=torch.float32)
    
    # Forward pass
    output = linear(x)
    torch_output = torch_linear(torch_x)
    
    # Loss
    loss = summation((output - y) ** 2)
    torch_loss = ((torch_output - torch_y) ** 2).sum()
    
    # Backward pass
    loss.backward()
    torch_loss.backward()
    
    # Optimizer step
    opt = Adam([linear.weight, linear.bias], lr=lr, beta1=0.9, beta2=0.999, eps=eps)
    torch_opt = torch.optim.Adam(torch_linear.parameters(), lr=lr, betas=betas, eps=eps)
    
    opt.step()
    torch_opt.step()
    
    # Check updated weights
    np.testing.assert_allclose(linear.weight.data, torch_linear.weight.data.numpy(), rtol=1e-4)
    np.testing.assert_allclose(linear.bias.data, torch_linear.bias.data.numpy(), rtol=1e-4)
    print("Adam: Parameter updates match PyTorch ✓")
    
    print("Adam test passed!")


if __name__ == "__main__":
    # Run the tests
    test_linear_fwd_bwd()
    test_sgd()
    test_adam()