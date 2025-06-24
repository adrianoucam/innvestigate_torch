from .base import Analyzer
import torch
import torch.nn.functional as F

class GradCAMAnalyzer(Analyzer):
    def __init__(self, model, target_layer):
        super().__init__(model)
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def analyze(self, input_tensor):
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1)
        self.model.zero_grad()
        output[0, target_class[0]].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam / cam.max()
        return cam