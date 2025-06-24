# innvestigate_torch

Uma reimplementação da biblioteca iNNvestigate usando PyTorch. Suporta diversos métodos de explicabilidade como:

- Gradient (Saliency)
- Integrated Gradients
- SmoothGrad
- LRP (Z-Rule, Epsilon, AlphaBeta)
- NoiseTunnel
- GradCAM

## Instalação

```bash
pip install .
```

## Exemplo de uso

```python
from analyzers.gradient import GradientAnalyzer
analyzer = GradientAnalyzer(model)
attribution = analyzer.analyze(input_tensor)
```
