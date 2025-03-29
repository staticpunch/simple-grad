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


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def __call__(self, x: Tensor) -> Tensor:
        output = x
        for module in self.modules:
            output = module(output)
        return output