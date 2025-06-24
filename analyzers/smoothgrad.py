from .base import Analyzer
import torch

class SmoothGradAnalyzer(Analyzer):
    def __init__(self, model, samples=25, noise_level=0.1):
        super().__init__(model)
        self.samples = samples
        self.noise_level = noise_level

    def analyze(self, input_tensor):
        grads = torch.zeros_like(input_tensor)
        for _ in range(self.samples):
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = (input_tensor + noise).clone().detach().requires_grad_(True)
            output = self.model(noisy_input)
            target_class = output.argmax(dim=1)
            for i, idx in enumerate(target_class):
                self.model.zero_grad()
                output[i, idx].backward(retain_graph=True)
                grads[i] += noisy_input.grad.data[i] if noisy_input.grad is not None else 0
        return grads / self.samples