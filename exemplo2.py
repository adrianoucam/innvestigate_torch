import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision.models import resnet50, ResNet50_Weights
from analyzers.gradient import GradientAnalyzer
from analyzers.smoothgrad import SmoothGradAnalyzer
from analyzers.lrp import LRPZRuleAnalyzer, LRPEpsilonRuleAnalyzer, LRPAlphaBetaAnalyzer
from analyzers.noise_tunnel import NoiseTunnelAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega modelo pré‑treinado
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights).to(device)
model.eval()

# Preprocessamento compatível com o modelo
preprocess = weights.transforms()

# Carrega CIFAR‑10 como exemplo, mas para ResNet ideal seria ImageNet
transform = transforms.Compose([ transforms.Resize((224,224)), transforms.ToTensor() ])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
classes = testset.classes

def plot_side_by_side(img_tensor, attributions, titles, label, prediction):
    fig, axes = plt.subplots(1, len(attributions)+1, figsize=(4*(len(attributions)+1),4))

    img = img_tensor.squeeze().permute(1,2,0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title(f"Input\nReal: {label}\nPred: {prediction}")
    axes[0].axis('off')

    for ax, attr, title in zip(axes[1:], attributions, titles):
        arr = attr.squeeze().detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.mean(np.abs(arr), axis=0)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        ax.imshow(img)
        ax.imshow(arr, cmap='jet', alpha=0.5)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


analyzers = [
    ("Gradient", GradientAnalyzer(model)),
    ("SmoothGrad", SmoothGradAnalyzer(model)),
    ("LRP‑Z", LRPZRuleAnalyzer(model)),
    ("LRP‑Epsilon", LRPEpsilonRuleAnalyzer(model)),
    ("LRP‑AlphaBeta", LRPAlphaBetaAnalyzer(model)),
    ("NoiseTunnel (Grad)", NoiseTunnelAnalyzer(model, GradientAnalyzer(model))),
]

for i, (inputs, labels) in enumerate(testloader):
    # Pré‑processa entrada para o modelo corretamente
    inputs = torch.stack([preprocess(transforms.ToPILImage()(inputs.squeeze()))]).to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    pred = outputs.argmax(dim=1).item()
    real = classes[labels.item()]
    pred_label = pred  # note: classe imagenet vs CIFAR mismatch

    atts, titles = [], []
    for name, analyzer in analyzers:
        try:
            attr = analyzer.analyze(inputs, index=pred)
        except TypeError:
            attr = analyzer.analyze(inputs)
        atts.append(attr)
        titles.append(name)
    plot_side_by_side(inputs, atts, titles, real, pred_label)

    if i >= 4:
        break
