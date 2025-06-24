from .base import Analyzer
import torch

class LRPZRuleAnalyzer(Analyzer):
    def analyze(self, input_tensor):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)
        grads = torch.zeros_like(input_tensor)
        for i, idx in enumerate(target_class):
            self.model.zero_grad()
            output[i, idx].backward(retain_graph=True)
            grads[i] = input_tensor.grad[i] * input_tensor[i]
        return grads

class LRPEpsilonRuleAnalyzer(Analyzer):
    def __init__(self, model, epsilon=1e-6):
        super().__init__(model)
        self.epsilon = epsilon

    def analyze(self, input_tensor):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)
        grads = torch.zeros_like(input_tensor)
        for i, idx in enumerate(target_class):
            self.model.zero_grad()
            output[i, idx].backward(retain_graph=True)
            z = input_tensor.grad[i] * input_tensor[i]
            grads[i] = z / (z.abs() + self.epsilon)
        return grads

class LRPAlphaBetaAnalyzer(Analyzer):
    def __init__(self, model, alpha=1.0, beta=0.0):
        super().__init__(model)
        self.alpha = alpha
        self.beta = beta

    def analyze(self, input_tensor):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)
        grads = torch.zeros_like(input_tensor)
        for i, idx in enumerate(target_class):
            self.model.zero_grad()
            output[i, idx].backward(retain_graph=True)
            g = input_tensor.grad[i]
            grads[i] = self.alpha * torch.clamp(g, min=0) * input_tensor[i] + \
                       self.beta * torch.clamp(g, max=0) * input_tensor[i]
        return grads