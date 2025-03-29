import numpy as np

from . import ops
from .tensor import Tensor

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _unpack_params(value: object):
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []
        
class Module:
    def parameters(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self): # List[Tensor]
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)
            
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.weight = Parameter(np.random.randn(out_features, in_features) * 0.01, requires_grad=True)
        self.bias = Parameter(np.zeros(out_features), requires_grad=True)
    
    def __call__(self, x):
        output = ops.matmul(x, self.weight.transpose())
        output = output + ops.broadcast_to(self.bias, output.shape)
        return output
        # return matmul(x, self.weight.transpose()) + self.bias
    
    def load_state_dict(self, state_dict):
        self.weight.data = state_dict["weight"].detach().numpy()
        self.bias.data = state_dict["bias"].detach().numpy()

    def state_dict(self):
        return dict(
            weight=self.weight.data,
            bias=self.bias.data,
        )

class ReLU(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Flatten(Module):
    def __call__(self, X):
        """
        input: 
            X: (B, X1, X2, ...)
        output:
            flattened X: (B, X1 * X2 * ...)
        """
        shape = X.shape 
        new_shape = (shape[0], np.prod(shape[1:]))
        return ops.reshape(X, new_shape)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def __call__(self, x: Tensor) -> Tensor:
        output = x
        for module in self.modules:
            output = module(output)
        return output

def one_hot(n_dim: int, y: Tensor) -> Tensor:
    """
    n_dim (int): Number of classes (length of each one-hot vector).
    y (Tensor): Tensor of shape (batch_size,) with integer class indices.
    """
    batch_size = y.shape[0]
    one_hot_np = np.zeros((batch_size, n_dim), dtype=int)
    for i in range(batch_size):
        one_hot_np.data[i, int(y.data[i])] = 1
    one_hot_tensor = Tensor(one_hot_np, requires_grad=False)
    return one_hot_tensor

class CrossEntropyLoss(Module):
    def __call__(self, logits: Tensor, y: Tensor):
        n_dim = logits.shape[-1] # logits.shape: (bsz, n_dim)
        y_one_hot = one_hot(n_dim, y) # y.shape: (bsz,)
        zy = ops.summation(logits * y_one_hot, axis=(1,))
        zy = ops.reshape(zy, (-1, 1))
        zy = ops.broadcast_to(zy, logits.shape)

        losses = ops.logsumexp(logits - zy, axis=(1))
        # total_loss = ops.summation(losses) / losses.shape[-1]
        total_loss = ops.summation(losses)
        norm_term = ops.broadcast_to(
            Tensor(np.array(losses.shape[-1]), requires_grad=False), 
            shape=total_loss.shape
        )
        total_loss = total_loss / norm_term
        return total_loss