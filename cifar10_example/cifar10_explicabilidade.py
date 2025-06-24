import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from analyzers.gradient import GradientAnalyzer
from analyzers.smoothgrad import SmoothGradAnalyzer
from analyzers.lrp import LRPZRuleAnalyzer, LRPEpsilonRuleAnalyzer, LRPAlphaBetaAnalyzer
from analyzers.noise_tunnel import NoiseTunnelAnalyzer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transformações e dados
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
classes = testset.classes

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("simple_cifar10_model.pth", map_location=device))
model.eval()

def plot_side_by_side(img_tensor, attributions, titles, label, prediction):
    num_methods = len(attributions)
    fig, axes = plt.subplots(1, num_methods + 1, figsize=(4 * (num_methods + 1), 4))
    
    img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    img = np.clip(img, 0, 1)
    axes[0].imshow(img)
    axes[0].set_title(f"Input\nReal: {label}\nPred: {prediction}")
    axes[0].axis('off')
    
    for i in range(num_methods):
        attr = attributions[i].squeeze()
        if attr.ndim == 3:
            attr = attr.permute(1, 2, 0).detach().cpu().numpy()
            attr = np.abs(attr).mean(axis=2)
        elif attr.ndim == 2:
            attr = attr.detach().cpu().numpy()
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        axes[i + 1].imshow(img)
        axes[i + 1].imshow(attr, cmap='jet', alpha=0.5)
        axes[i + 1].set_title(titles[i])
        axes[i + 1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Lista de explicadores
analyzers = [
    ("Gradient", GradientAnalyzer(model)),
    ("SmoothGrad", SmoothGradAnalyzer(model)),
    ("LRP-Z", LRPZRuleAnalyzer(model)),
    ("LRP-Epsilon", LRPEpsilonRuleAnalyzer(model)),
    ("LRP-AlphaBeta", LRPAlphaBetaAnalyzer(model)),
    ("NoiseTunnel (Grad)", NoiseTunnelAnalyzer(model, GradientAnalyzer(model))),
]

# Executar para as primeiras 5 imagens
for i, (inputs, labels) in enumerate(testloader):
    inputs = inputs.to(device)
    inputs.requires_grad = True

    outputs = model(inputs)
    pred = torch.argmax(outputs, dim=1)
    real_label = classes[labels.item()]
    pred_label = classes[pred.item()]

    attributions = []
    titles = []

    for name, analyzer in analyzers:
        attr = analyzer.analyze(inputs)
        attributions.append(attr)
        titles.append(name)

    plot_side_by_side(inputs, attributions, titles, label=real_label, prediction=pred_label)

    if i == 4:
        break
