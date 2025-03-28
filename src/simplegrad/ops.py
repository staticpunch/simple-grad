import numpy as np
import torch

from .tensor import Tensor


def relu(tensor):
    out = Tensor(np.maximum(0, tensor.data), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += (tensor.data > 0) * out.grad
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def sigmoid(tensor):
    out = Tensor(1 / (1 + np.exp(-tensor.data)), requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            tensor.grad += out.data * (1 - out.data) * out.grad
    
    out._backward = _backward
    out._prev = {tensor, }
    return out


def matmul(tensor1, tensor2):
    assert isinstance(tensor2, Tensor), "Operand must be a Tensor"
    out = Tensor(tensor1.data @ tensor2.data, requires_grad=tensor1.requires_grad or tensor2.requires_grad)
    
    def _backward():
        if tensor1.requires_grad:
            tensor1.grad += out.grad @ tensor2.data.T
        if tensor2.requires_grad:
            tensor2.grad += tensor1.data.T @ out.grad
    
    out._backward = _backward
    out._prev = {tensor1, tensor2}
    return out

def log(tensor):
    out = Tensor(np.log(tensor.data), requires_grad=tensor.requires_grad)
    def _backward():
        tensor.grad += out.grad / tensor.data
    out._backward = _backward
    out._prev = {tensor, }
    return out
    
def logsumexp(tensor, axis=None, keepdims=False):
    """
    mathematical operations, applied to 1D vector: 
    forward: log(e^z1 + e^z2 + ... + e^zn) = sum(e^zi)
    backward: local_grad[i] = e^zi / sum(e^zi)
    ------
    for numerical stability:
    forward: log(sum(e^zi))  = log(sum(e^(zi - zmax)) * e^zmax)
                             = log(sum(e^(zi - zmax))) + log(e^zmax)
    backward: e^zi/sum(e^zi) = e^(zi - zmax) / sum(e^(zi - zmax))
    """
    max_z = np.max(tensor.data, axis=axis, keepdims=True)
    stable_z = tensor.data - max_z
    exp_stable_z = np.exp(stable_z)
    stable_sum = np.sum(exp_stable_z, axis=axis, keepdims=keepdims)
    max_term = max_z if keepdims else np.squeeze(max_z, axis=axis)
    data = np.log(stable_sum) + max_term
    out = Tensor(data, requires_grad=tensor.requires_grad)
    
    def _backward():
        if tensor.requires_grad:
            if axis is None:
                # For None axis, basically all dims.
                if not keepdims:
                    grad_shaped = out.grad * np.ones_like(tensor.data)
                    softmax_terms = exp_stable_z / np.sum(exp_stable_z)
                    tensor.grad += grad_shaped * softmax_terms
                else:
                    # keepdims=True with axis=None
                    softmax_terms = exp_stable_z / stable_sum
                    tensor.grad += out.grad * softmax_terms
            else:
                # For specific axis reduction
                grad_shaped = out.grad
                if not keepdims:
                    grad_shaped = np.expand_dims(grad_shaped, axis=axis)
    
                denom = stable_sum if keepdims \
                        else np.expand_dims(stable_sum, axis=axis)
                softmax_terms = exp_stable_z / denom
                tensor.grad += grad_shaped * softmax_terms
    
    out._backward = _backward
    out._prev = {tensor, }
    return out
