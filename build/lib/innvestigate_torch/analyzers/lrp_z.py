
import torch
from .base import Analyzer

class LRPZ(Analyzer):
    def __init__(self, model, epsilon=1e-6):
        super().__init__(model)
        self.epsilon = epsilon

    def analyze(self, input_tensor, target=None):
        x = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(x)

        if target is None:
            target = output.argmax(dim=1)
        one_hot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
        relevance = (output * one_hot).sum()

        self.model.zero_grad()
        relevance.backward(retain_graph=True)

        relevance_map = x.grad * x  # LRP-Z basic relevance rule
        return relevance_map
