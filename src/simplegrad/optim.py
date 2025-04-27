import numpy as np
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data) if p.requires_grad else None


class NaiveSGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) with optional L2 regularization.

    - SGD minimizes a loss by updating parameters opposite to the gradient.
    - L2-regularized loss: \( L_{\text{reg}}(w) = L(w) + \frac{\lambda}{2} \|w\|_2^2 \)
      - \( \|w\|_2^2 \): squared L2 norm of parameters.
      - \( \lambda \): regularization strength (`weight_decay`).
    - Gradient: \( \nabla L_{\text{reg}}(w) = \nabla L(w) + \lambda w \)
    - Update rule: \( w \leftarrow w - \eta (\nabla L(w) + \lambda w) \)
      - \( \eta \): learning rate (`lr`).
    - Implementation: adds \( \lambda w \) to gradient as weight decay.
    """

    def __init__(self, params, lr=0.01, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for w in self.params:
            gradient = w.grad + self.weight_decay * w.data
            w.data = w.data - self.lr * gradient


class Adam(Optimizer):
    """
    Adam. I followed the recipe in the paper https://arxiv.org/pdf/1412.6980
    """
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad
            # Initialize state for this parameter if not already done
            if i not in self.state:
                self.state[i] = {
                    'm': np.zeros_like(p.data), # 1st moment vector
                    'v': np.zeros_like(p.data), # 2nd moment vector
                    't': 0                      # Timestep
                }
            state = self.state[i]

            # Apply weight decay (L2 regularization) before momentum calculation
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data

            # Increment timestep
            state['t'] += 1
            t = state['t']
            m, v = state['m'], state['v']
            beta1, beta2 = self.beta1, self.beta2
            m = beta1 * m + (1 - beta1) * grad
            state['m'] = m
            v = beta2 * v + (1 - beta2) * np.square(grad)
            state['v'] = v
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            p.data -= update
            