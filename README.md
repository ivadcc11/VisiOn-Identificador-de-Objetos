# 🔍 Detector de Objetos - Vision AI

## Objetivo

Aplicação web em Python que identifica objetos em imagens usando inteligência artificial (YOLOv8). O usuário pode enviar uma foto ou tirar uma foto com a webcam, e o sistema mostra quais objetos foram encontrados e quantos de cada tipo. Pode ser usado para contar itens em uma foto ou analisar objetos em tempo real.

## Tecnologias Utilizadas

- **Python**: 3.10 a 3.13 (testado)
- **[Streamlit](https://streamlit.io/)**: Interface web interativa
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/)**: Modelo de detecção de objetos
- **OpenCV**: Leitura e processamento de imagens
- **Pillow**: Manipulação de imagens
- **NumPy**: Operações numéricas

## Funcionalidades

- [x] Carregamento do modelo YOLOv8 com ajuste de nível de confiança
- [x] Upload de imagem com detecção e resumo dos objetos
- [x] Captura de foto pela webcam com detecção em tempo real
- [x] Menu de configurações (seleção de fonte de imagem, ajuste de confiança)
- [x] Listagem detalhada de objetos detectados com contagem

## Pré-requisitos

- **Python 3.10** ou superior (testado até Python 3.13)
- **Windows** (recomendado para desenvolvimento)
- Conexão com a internet (para download do modelo YOLOv8 na primeira execução)

## Guia de Instalação

### 1. Clonar o Repositório

```bash
git clone https://github.com/ivadcc11/VisiOn-Identificador-de-Objetos.git
cd VisiOn-Identificador-de-Objetos
```

### 2. Instalar as Dependências

Com o Python configurado (3.10-3.13), execute:

```bash
pip install -r requirements.txt
```

Isso instalará automaticamente:
- Streamlit
- Ultralytics (YOLO)
- OpenCV
- Torch e TorchVision
- NumPy, Pandas e Pillow

### 3. Executar a Aplicação

Execute o aplicativo usando o Streamlit:

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador em **http://localhost:8501**.

> **Nota:** Na primeira execução, o modelo `yolov8n.pt` será baixado automaticamente. Certifique-se de ter conexão com a internet.

## Como Usar

1. **Upload de Arquivo:** Selecione "Upload de Arquivo" na barra lateral e envie uma imagem (JPG, JPEG, PNG ou WebP)
2. **Webcam:** Selecione "Webcam" e tire uma foto para análise
3. **Ajuste de Confiança:** Use o controle deslizante para ajustar o nível mínimo de confiança das detecções
4. **Resultado:** Visualize a imagem processada e o resumo dos objetos identificados

## Integrantes

- **Davi Camargo Campos** (Desenvolvedor)

## Observações

- A webcam do navegador funciona melhor em `localhost` ou conexões HTTPS
- O projeto foi desenvolvido para detecção estática de objetos em imagens e fotos via webcam
- O modelo YOLOv8 nano é leve e otimizado para detecção rápida em CPU
- Créditos: Modelo YOLOv8 desenvolvido pela [Ultralytics](https://www.ultralytics.com/)
