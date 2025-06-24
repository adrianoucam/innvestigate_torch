
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch
from innvestigate_torch import LRPZ

# Modelo
model = resnet18(pretrained=True)
analyzer = LRPZ(model)

# Pré-processamento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = transform(Image.open("example.jpg")).unsqueeze(0)
img = img.to("cuda" if torch.cuda.is_available() else "cpu")

# Relevância
relevance = analyzer.analyze(img)
print("Relevance shape:", relevance.shape)
