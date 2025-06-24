from .base import Analyzer
import torch

class GradientAnalyzer(Analyzer):
    def analyze(self, input_tensor):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)
        grads = torch.zeros_like(input_tensor)
        for i, idx in enumerate(target_class):
            self.model.zero_grad()
            output[i, idx].backward(retain_graph=True)
            grads[i] = input_tensor.grad.data[i]
        return grads