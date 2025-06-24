from .base import Analyzer
import torch

class IntegratedGradientsAnalyzer(Analyzer):
    def __init__(self, model, steps=50):
        super().__init__(model)
        self.steps = steps

    def analyze(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        scaled_input = (input_tensor * alpha).detach().requires_grad_()
        grads = torch.zeros_like(input_tensor)
        for scaled_input in scaled_inputs:
            scaled_input.requires_grad = True
            output = self.model(scaled_input)
            target_class = output.argmax(dim=1)
            for i, idx in enumerate(target_class):
                self.model.zero_grad()
                output[i, idx].backward(retain_graph=True)
                grads[i] += scaled_input.grad.data[i] if scaled_input.grad is not None else 0
        return (input_tensor - baseline) * (grads / self.steps)