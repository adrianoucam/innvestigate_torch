import sys
sys.path.append(r"C:\Users\adria\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages")



import os
import csv
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# ✅ Importando da biblioteca instalada via pip
from innvestigate_torch.analyzers.gradient import GradientAnalyzer
from innvestigate_torch.analyzers.integrated_gradients import IntegratedGradientsAnalyzer
from innvestigate_torch.analyzers.smoothgrad import SmoothGradAnalyzer
from innvestigate_torch.analyzers.lrp import (
    LRPZRuleAnalyzer,
    LRPEpsilonRuleAnalyzer,
    LRPAlphaBetaAnalyzer
)
from innvestigate_torch.analyzers.noise_tunnel import NoiseTunnelAnalyzer
from innvestigate_torch.analyzers.gradcam import GradCAMAnalyzer


def load_image(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    return image


def save_explanation(image_tensor, attribution, name, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img = image_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    attr = attribution.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    attr_map = np.abs(attr).mean(axis=2) if attr.ndim == 3 else attr.squeeze()
    attr_map = attr_map - attr_map.min()
    attr_map = attr_map / (attr_map.max() + 1e-8)

    # Salvar imagem sobreposta
    plt.imshow(img * 0.2 + 0.5)
    plt.imshow(attr_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(f"{output_dir}/{name}.png", bbox_inches='tight')
    plt.close()

    # Salvar CSV
    csv_path = f"{output_dir}/{name}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in attr_map:
            writer.writerow(row)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True).to(device)

    # Usar argumento da linha de comando ou padrão
    image_path = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"

    try:
        image_tensor = load_image(image_path, device)
    except Exception as e:
        print(f"Erro ao carregar a imagem: {e}")
        sys.exit(1)

    analyzers = [
        ("gradient", GradientAnalyzer(model)),
        ("integrated_gradients", IntegratedGradientsAnalyzer(model)),
        ("smoothgrad", SmoothGradAnalyzer(model)),
        ("lrp_zrule", LRPZRuleAnalyzer(model)),
        ("lrp_epsilon", LRPEpsilonRuleAnalyzer(model)),
        ("lrp_alphabeta", LRPAlphaBetaAnalyzer(model)),
        ("noise_tunnel", NoiseTunnelAnalyzer(model, GradientAnalyzer(model))),
        ("gradcam", GradCAMAnalyzer(model, model.layer4[-1]))
    ]

    for name, analyzer in analyzers:
        print(f">>> Executando {name}...")
        attribution = analyzer.analyze(image_tensor)
        save_explanation(image_tensor, attribution, name)
        print(f"✔ Salvo: outputs/{name}.png e outputs/{name}.csv")
