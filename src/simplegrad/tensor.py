import numpy as np
import torch


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    @staticmethod
    def from_torch(tensor):
        return Tensor(tensor.detach().cpu().numpy(), requires_grad=tensor.requires_grad)

    def to_torch(self):
        return torch.tensor(self.data, requires_grad=self.requires_grad,
                            dtype=torch.float32)

    def transpose(self):
        out = Tensor(np.swapaxes(self.data, -1, -2), requires_grad=self.requires_grad) 

        def _backward():
            if self.requires_grad:
                self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward 
        out._prev = {self, }
        return out 
    
    def __add__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            """
            out = a - b
            a.grad = out.grad * (∂_out / ∂_a)
                   = out.grad * (∂_(a - b) / ∂_a)
                   = out.grad * 1
            b.grad = out.grad * (∂_out / ∂_b)
                   = out.grad * (∂_(a - b) / ∂_b)
                   = out.grad * (-1)
            """
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad * -1
                
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __truediv__(self, other):
        assert isinstance(other, Tensor), "Operand must be a Tensor"
        out = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            """
            out = a / b
            a.grad = out.grad * (∂_out / ∂_a)
                   = out.grad * (∂_(a / b) / ∂_a)
                   = out.grad * (1 / b)
            b.grad = out.grad * (∂_out / ∂_b)
                   = out.grad * (∂_(a / b) / ∂_b)
                   = out.grad * (-a / b**2)
            """
            if self.requires_grad:
                self.grad += out.grad * (1 / other.data)
            if other.requires_grad:
                other.grad += out.grad * (-self.data / other.data**2)
                
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __neg__(self):
        out = Tensor(-self.data, requires_grad=self.requires_grad)

        def _backward():
            """
            out = -a
            a.grad = -1
            """
            if self.requires_grad:
                self.grad += out.grad * -1
                
        out._backward = _backward
        out._prev = {self}
        return out

    def __pow__(self, other):
        from numbers import Number
        assert isinstance(other, (int, float)), "Operand must be a Scalar"
        out = Tensor(self.data ** other, requires_grad=self.requires_grad)

        def _backward():
            """
            out = a ** b
            a.grad = b * (a ** (b - 1))
            """
            if self.requires_grad:
                local_grad = other * (self.data ** (other - 1))
                self.grad += out.grad * local_grad
                
        out._backward = _backward
        out._prev = {self}
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for tensor in reversed(topo):
            tensor._backward()
