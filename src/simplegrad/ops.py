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

def summation(tensor, axis=None, keepdims=False):
    """
    local_grad = d.sum(x) / d.xi = 1.
    therefore derivative of x is basically just out.grad
    broadcasted to the shape of the input tensor.
    """
    out = Tensor(np.sum(tensor.data, axis=axis, keepdims=keepdims), requires_grad=tensor.requires_grad)
    def _backward():
        if tensor.requires_grad:
            input_shape, axes = tensor.data.shape, axis
            
            if not keepdims:
                if axis is None: # if self.axes is None, take sum over all axes.
                    axes = tuple(i for i in range(len(input_shape)))
                elif isinstance(axis, int): 
                    axes = (axis,)

                shape_range = range(len(input_shape))
                mask = np.array([0 if i in axes else 1 for i in shape_range])
                new_shape = np.array(input_shape) * mask + (1 - mask)
                grad = np.reshape(out.grad, new_shape)
                grad = np.broadcast_to(grad, input_shape)
            else:
                grad = np.broadcast_to(out.grad, input_shape)
                
            tensor.grad += grad
        
    out._backward = _backward
    out._prev = {tensor,}
    return out
    
def broadcast_to(tensor, shape):
    """this is interestingly the reverse of summation."""
    if tensor.shape == shape: # Optimization: no-op if shapes match
        return tensor
        
    out_data = np.broadcast_to(tensor.data, shape)
    out = Tensor(out_data, requires_grad=tensor.requires_grad)
    
    input_shape = tensor.shape # Capture input shape for backward pass
    def _backward():
        if tensor.requires_grad:
            ishape, oshape = tensor.data.shape, out.grad.shape
            ## in = (3, 1, 4), out = (3, 5, 4) -> aligned = (3, 1, 4)
            ## i think numpy only implicitly broadcast to prefix dims :/
            aligned = [1] * (len(oshape) - len(ishape)) + list(ishape)
            broadcast_axes = tuple([i for i, axis in enumerate(aligned) if axis == 1])
            grad = np.sum(out.grad, axis=broadcast_axes, keepdims=True)
            grad = np.reshape(grad, ishape)

            tensor.grad += grad
        
    out._backward = _backward
    out._prev = {tensor,}
    return out

def softmax(tensor, axis: int = None):
    """
    to reduce headache, actually I should implement an exp ops,
    then let the chain rule do its job automatically.
    """
    lse = logsumexp(tensor, axis=axis, keepdims=True)
    # print(tensor.shape, lse.shape)
    lse_broadcast = broadcast_to(lse, tensor.data.shape)
    log_softmax = tensor - lse_broadcast
    out = exp(log_softmax)

    return out