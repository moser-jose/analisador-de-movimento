import subprocess
import sys
import os

def install_dependencies():
    """Instala as dependências necessárias com versões específicas."""
    print("Instalando dependências...")
    
    # Lista de dependências com versões fixas para garantir compatibilidade
    dependencies = [
        "python-dotenv",
        "ml-dtypes",
        "tensorflow",
        "ultralytics",
        "mediapipe"
    ]
    
    # Instalar cada dependência
    for dep in dependencies:
        print(f"Instalando {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("Todas as dependências foram instaladas com sucesso!")

def test_imports():
    """Testa a importação das principais bibliotecas."""
    print("Testando imports...")
    
    try:
        import numpy
        print("✓ NumPy importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar NumPy: {e}")
    
    try:
        import cv2
        print(f"✓ OpenCV importado com sucesso (versão {cv2.__version__})")
    except ImportError as e:
        print(f"✗ Erro ao importar OpenCV: {e}")
    
    try:
        from ml_dtypes import float8_e4m3fn, float8_e5m2
        print(f"✓ ML-DTtypes importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar ML-DTtypes: {e}")
    except AttributeError:
        print(f"✓ ML-DTtypes importado com sucesso (versão desconhecida)")
    
    try:
        import mediapipe
        print(f"✓ MediaPipe importado com sucesso (versão {mediapipe.__version__})")
    except ImportError as e:
        print(f"✗ Erro ao importar MediaPipe: {e}")
    
    try:
        import tensorflow
        print(f"✓ TensorFlow importado com sucesso (versão {tensorflow.__version__})")
    except ImportError as e:
        print(f"✗ Erro ao importar TensorFlow: {e}")

    try:
        import tflite_runtime
        print(f"✓ tflite_runtime importado com sucesso (versão {tflite_runtime.__version__})")
    except ImportError as e:
        print(f"✗ Erro ao importar tflite_runtime: {e}")
    
    try:
        from ultralytics import YOLO
        print("✓ YOLO importado com sucesso")
    except ImportError as e:
        print(f"✗ Erro ao importar YOLO: {e}")

if __name__ == "__main__":
    print("=== Configuração do Analisador de Movimento com YOLO e MediaPipe ===")
    
    # Perguntar se o usuário quer instalar as dependências
    install = input("Deseja instalar as dependências? (s/n): ").lower() == 's'
    if install:
        install_dependencies()
    
    # Testar imports independentemente da escolha anterior
    test_imports()
    
    print("\nSe todos os imports foram bem-sucedidos, você pode executar o programa com:")
    print("python main.py --engine yolo")
    print("ou")
    print("python main.py --engine mediapipe")
    print("\nAlternativamente, você pode usar o analisador independente:")
    print("python mediapipe_analyzer.py") 