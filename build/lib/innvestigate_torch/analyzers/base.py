
import torch

class Analyzer:
    def __init__(self, model):
        self.model = model.eval()

    def analyze(self, input_tensor, target=None):
        raise NotImplementedError("Each analyzer must implement the analyze() method.")
