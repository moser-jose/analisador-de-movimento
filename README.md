# Analisador de Movimento com YOLO

Este projeto implementa um analisador de movimento em tempo real usando o modelo YOLO para detecção de pose. O sistema é capaz de detectar e analisar movimentos do corpo humano, calculando ângulos das articulações e fornecendo feedback visual.

## Funcionalidades

- Detecção de pose em tempo real usando YOLO
- Cálculo de ângulos das articulações
- Visualização em tempo real dos ângulos
- Suporte para processamento em GPU (via TFLite)
- Salvamento de dados de análise
- Interface gráfica com gráficos de ângulos

## Requisitos

- Python 3.9+
- CUDA (opcional, para aceleração GPU)
- Webcam ou câmera IP

## Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/analisador-movimento.git
cd analisador-movimento
```

2. Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração

O projeto usa seguintes variáveis de ambiente para configuração:

```env
CAMERA_DEVICE=0
CAMERA_FPS=60
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
MODEL_PATH=yolo11n-pose.pt
TFLITE_PATH=model_tflite.tflite
CONFIDENCE_THRESHOLD=0.5
SAVE_DATA=true
OUTPUT_DIR=dados_dos_movimentos
BUFFER_SIZE=30 
```

## Uso

Execute o programa:

```bash
python main.py
```

### Controles

- Pressione 'q' para sair
- Os dados de análise são salvos automaticamente ao encerrar

## Estrutura do Projeto

```
analisador-movimento/
├── config.py           # Configurações do projeto
├── main.py             # Código principal
├── requirements.txt    # Dependências
└── README.md           # Documentação
```

## Contribuição

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.

