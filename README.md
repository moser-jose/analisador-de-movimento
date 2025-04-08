# Analisador de Movimento com YOLO e MediaPipe

Este projeto implementa um analisador de movimento em tempo real usando o modelo YOLO para detecção de pose e MediaPipe como ferramenta adicional para cálculo preciso de ângulos das articulações. O sistema é capaz de detectar e analisar movimentos do corpo humano, calculando ângulos das articulações e fornecendo feedback visual.

## Funcionalidades

- Detecção de pose em tempo real usando YOLO ou MediaPipe
- Cálculo de ângulos das articulações 
- Visualização em tempo real dos ângulos
- Suporte para processamento em GPU (via TFLite)
- Salvamento de dados de análise
- Interface gráfica com gráficos de ângulos
- Opção de escolher entre YOLO e MediaPipe para cálculo de ângulos

## Requisitos

- Python 3.9+
- CUDA (opcional, para aceleração GPU)
- Webcam ou câmera IP
- MediaPipe
- OpenCV

## Instalação

1.1. Clone o repositório via https:

```bash
git clone https://github.com/moser-jose/analisador-de-movimento.git
cd analisador-movimento
```

1.2. Clone o repositório via ssh:
```bash
git clone git@github.com:moser-jose/analisador-de-movimento.git
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
# Usando o script de configuração (recomendado)
python setup.py

# ou manualmente
pip install -r requirements.txt
```

O script de configuração irá instalar as versões corretas das dependências para evitar incompatibilidades, principalmente com MediaPipe e TensorFlow.

## Configuração

O projeto usa as seguintes variáveis de ambiente para configuração:

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
ENGINE=yolo  # ou mediapipe
```

## Uso

4. Execute o programa especificando o motor de análise desejado:

```bash
# Usando YOLO para detecção e cálculo de ângulos
python main.py --engine yolo

# Usando o MediaPipe para detecção e cálculo de ângulos
python main.py --engine mediapipe
```

Se nenhum motor for especificado, o programa usará o valor definido na variável de ambiente `ENGINE` ou YOLO como padrão.

### Controles

- Pressione 'q' para sair
- Os dados de análise são salvos automaticamente ao encerrar, incluindo o nome do motor utilizado no nome do arquivo

## Contribuição

Contribuições são bem-vindas! Por favor, sinta-se à vontade para submeter pull requests.
