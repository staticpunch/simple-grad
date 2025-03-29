import simplegrad as sg
import simplegrad.module as snn
from simplegrad import Tensor, ops
from simplegrad.module import Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def test_cross_entropy_loss():
    print("=== TESTING CROSS ENTROPY LOSS ===")
    
    # More challenging test case with extreme values
    batch_size = 32
    num_classes = 16
    
    # Random logits with mix of extreme values
    # np.random.seed(42)
    logits_data = np.random.randn(batch_size, num_classes) * 20
    # logits_data[0:5, :] *= 1e10  # Very large values
    # logits_data[5:10, :] *= 1e-10  # Very small values
    # logits_data[10:15, :] = np.random.choice([-1e15, 1e15], size=(5, num_classes))  # Mixed extremes
    
    # Random labels
    labels_data = np.random.randint(0, num_classes, size=batch_size).astype(np.float32)
    
    # Create tensors
    logits = Tensor(logits_data, requires_grad=True)
    labels = Tensor(labels_data, requires_grad=False)
    
    # Create PyTorch tensors for comparison
    pt_logits = torch.tensor(logits_data, requires_grad=True)
    pt_labels = torch.tensor(labels_data, dtype=torch.long)  # PyTorch expects long for labels
    
    # Compute loss with our implementation
    loss_fn = snn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    
    # Compute loss with PyTorch
    pt_loss = F.cross_entropy(pt_logits, pt_labels, reduction='mean')
    
    print(f"Our implementation loss: {loss.data}")
    print(f"PyTorch loss: {pt_loss.item()}")
    
    # Check if losses are close
    print("Loss difference:", loss.data - pt_loss.item())
    if np.isclose(loss.data, pt_loss.item(), rtol=1e-4):
        print("✓ Forward pass successful - losses match!")
    else:
        print(f"✗ Forward pass failed - losses don't match. Difference: {np.abs(loss.data - pt_loss.item())}")
    
    # Test backward pass
    loss.backward()
    pt_loss.backward()
    
    # Check if gradients are close
    np.testing.assert_allclose(
        logits.grad,
        pt_logits.grad.numpy(),
        rtol=1e-4, atol=1e-5,
        err_msg="Gradients don't match"
    )
    print("✓ Backward pass successful - gradients match!")

if __name__ == "__main__":
    test_cross_entropy_loss()