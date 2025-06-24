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


O arquivo example.py pode ser usado depois que a biblioteca for instalada


# innvestigate_torch

**innvestigate_torch** é uma reimplementação da biblioteca [iNNvestigate](https://github.com/albermax/innvestigate) voltada para interpretabilidade de redes neurais, mas agora compatível com PyTorch. A biblioteca oferece múltiplos métodos de explicação que ajudam a entender o comportamento interno de modelos de deep learning.

## Métodos de Explicabilidade Suportados

### 🔹 Gradient (Saliency)
Baseado nos gradientes da saída em relação à entrada. Este método destaca quais pixels mais influenciam a previsão final. É simples e rápido, mas pode ser sensível a ruídos.

### 🔹 Integrated Gradients *(removido do example.py por instabilidade em backpropagation de variáveis não-folhas)*
Esse método estima a contribuição dos pixels interpolando entre uma baseline (entrada nula) e a entrada real. Fornece atribuições mais estáveis do que o gradiente puro.

### 🔹 SmoothGrad
Combina múltiplas execuções do método de gradiente com pequenas perturbações (ruído gaussiano) na entrada. O objetivo é reduzir ruídos e destacar regiões mais relevantes da imagem.

### 🔹 LRP (Layer-wise Relevance Propagation)
Explica a decisão do modelo redistribuindo o valor da saída (relevância) para os neurônios da entrada, camada por camada. Variantes implementadas:
- **Z-Rule**: redistribui relevância com base em ativações positivas e pesos.
- **Epsilon Rule**: adiciona um pequeno termo para evitar divisões por zero e ruídos.
- **AlphaBeta Rule**: balanceia contribuições positivas (α) e negativas (β).

### 🔹 NoiseTunnel
Uma extensão do método base (ex: Gradient ou IG) que adiciona ruído na entrada várias vezes e agrega os resultados para suavizar e destacar padrões consistentes. Suporta:
- `smoothgrad`
- `smoothgrad_sq`

### 🔹 GradCAM
Funciona com modelos CNN e utiliza os gradientes das últimas camadas convolucionais para gerar mapas de ativação, destacando regiões mais importantes da imagem para a predição.

---

## Instalação

```bash
pip install .


```


Exemplo 1 <br>
![Texto Alternativo](Figura_1.png)
<br>
Exemplo 2<br>
![Texto Alternativo](Figura_2.png)
<br>
Exemplo 3<br>
![Texto Alternativo](Figura_3.png)
<br>
Exemplo 4<br>
![Texto Alternativo](Figura_4.png)
<br>

Exemplo 5<br>
![Texto Alternativo](Figura_5.png)

<br>
Exemplo 6<br>
![Texto Alternativo](Figura_6.png)

<br>
Exemplo 7<br>
![Texto Alternativo](Figura_7.png)


