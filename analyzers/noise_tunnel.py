from .base import Analyzer
import torch

class NoiseTunnelAnalyzer(Analyzer):
    def __init__(self, model, base_analyzer, samples=10, noise_level=0.1):
        super().__init__(model)
        self.base_analyzer = base_analyzer
        self.samples = samples
        self.noise_level = noise_level

    def analyze(self, input_tensor):
        accumulated = torch.zeros_like(input_tensor)
        for _ in range(self.samples):
            noise = torch.randn_like(input_tensor) * self.noise_level
            noisy_input = input_tensor + noise
            attribution = self.base_analyzer.analyze(noisy_input)
            accumulated += attribution
        return accumulated / self.samples