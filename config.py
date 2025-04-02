import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da câmera
CAMERA_DEVICE = int(os.getenv('CAMERA_DEVICE', '0'))
CAMERA_FPS = int(os.getenv('CAMERA_FPS', '60'))
CAMERA_WIDTH = int(os.getenv('CAMERA_WIDTH', '1280'))
CAMERA_HEIGHT = int(os.getenv('CAMERA_HEIGHT', '720'))

# Configurações do modelo

MODEL_PATH = os.getenv('MODEL_PATH', 'YOLO11n-pose.pt')
TFLITE_PATH = os.getenv('TFLITE_PATH', 'model_tflite.tflite')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))

# Configurações de dados
SAVE_DATA = os.getenv('SAVE_DATA', 'true').lower() == 'true'
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'dados_dos_movimentos')
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', '30'))

# Criar diretório de saída se não existir
if SAVE_DATA:
    os.makedirs(OUTPUT_DIR, exist_ok=True) 