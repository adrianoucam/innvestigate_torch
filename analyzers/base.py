import torch.nn as nn
from abc import ABC, abstractmethod

class Analyzer(ABC):
    def __init__(self, model: nn.Module):
        self.model = model.eval()

    @abstractmethod
    def analyze(self, input_tensor):
        pass