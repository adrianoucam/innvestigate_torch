from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyzers.gradient import GradientAnalyzer
from analyzers.smoothgrad import SmoothGradAnalyzer
from analyzers.lrp import LRPZRuleAnalyzer, LRPEpsilonRuleAnalyzer, LRPAlphaBetaAnalyzer
from analyzers.noise_tunnel import NoiseTunnelAnalyzer
from analyzers.gradcam import GradCAMAnalyzer

def load_image(path, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image

def show_explanation(image_tensor, attribution):
    img = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # Tratamento para o mapa de ativação
    attr = attribution.squeeze()
    if attr.ndim == 3:
        attr = attr.permute(1, 2, 0).detach().cpu().numpy()
        attr = np.abs(attr).mean(axis=2)
    elif attr.ndim == 2:
        attr = attr.detach().cpu().numpy()
    else:
        raise ValueError(f"Atributo com número de dimensões inesperado: {attr.ndim}")

    # Visualização com sobreposição
    plt.imshow(img * 0.2 + 0.5)
    plt.imshow(attr, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)
    image_path = "example.jpg"
    image_tensor = load_image(image_path, device)

    analyzers = [
        GradientAnalyzer(model),
        SmoothGradAnalyzer(model),
        LRPZRuleAnalyzer(model),
        LRPEpsilonRuleAnalyzer(model),
        LRPAlphaBetaAnalyzer(model),
        NoiseTunnelAnalyzer(model, GradientAnalyzer(model)),
        GradCAMAnalyzer(model, model.layer4[-1])
    ]

    for analyzer in analyzers:
        attribution = analyzer.analyze(image_tensor)
        show_explanation(image_tensor, attribution)
