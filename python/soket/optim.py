from typing import List, Tuple
import soket


class Optimizer:
    def __init__(self, params: List[soket.Tensor]):
        self.params = params
    
    def step(self):
        raise NotImplementedError()
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None


# Stochastic variant of Gradient Descent with (nesterov) momentum.
class SGD(Optimizer):
    def __init__(
        self,
        params: List[soket.Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False
    ):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize

        # No initial momentum
        self.u = [ None ] * len(params)
    
    def step(self):
        for i, p in enumerate(self.params):            
            if p.grad is None: 
                continue

            if self.weight_decay != 0.0:
                p.grad = p.grad.data + self.weight_decay * p.data
            
            if self.momentum != 0.0:
                if self.u[i] is None:
                    self.u[i] = p.grad.data
                else:
                    self.u[i] = self.momentum * self.u[i].data +    \
                        (1 - self.dampening) * p.grad.data
                    
                if self.nesterov is True:
                    p.grad = p.grad.data + self.momentum * self.u[i].data
                else:
                    p.grad = self.u[i]
            
            if self.maximize is True:
                p.data = p.data + self.lr * p.grad.data
            else:
                p.data = p.data - self.lr * p.grad.data


class Adam(Optimizer):
    def __init__(
        self,
        params: List[soket.Tensor],
        lr: float = 0.001,
        betas: Tuple[float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        maximize: bool = False
    ):
        super().__init__(params)

        assert len(betas) > 1, 'Expected valid tuple of betas'
        for i in betas:
            assert isinstance(i, float), 'Betas must be floats'

        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
    
        # Store iteration
        self.t = 1

        # Store first and second momentum
        self.u = [ None ] * len(params)
        self.v = [ None ] * len(params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            if self.weight_decay != 0.0:
                p.grad = p.grad.data + self.weight_decay * p.data

            if self.maximize is True:
                p.grad = -p.grad.data
            
            if self.u[i] is None:
                self.u[i] = (1 - self.beta1) * p.grad.data
            else:
                self.u[i] = self.beta1 * self.u[i].data +    \
                    (1 - self.beta1) * p.grad.data
                
            if self.v[i] is None:
                self.v[i] = (1 - self.beta2) * (p.grad.data ** 2)
            else:
                self.v[i] = self.beta2 * self.v[i].data +    \
                    (1 - self.beta2) * (p.grad.data ** 2)
            
            # Biasing
            u = self.u[i].data / (1 - (self.beta1 ** self.t))
            v = self.v[i].data / (1 - (self.beta2 ** self.t))

            p.data = p.data - self.lr * u.data / ((v.data ** (1/2)) + self.eps)

        self.t += 1
