import cv2
import numpy as np
from ultralytics import YOLO
import config
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt

class MovementAnalyzer:
    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        self.cap = cv2.VideoCapture(config.CAMERA_DEVICE)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        self.angle_buffer = []
        self.start_time = time.time()
        
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def process_frame(self, frame):
        """Processa um único quadro e detecta poses."""
        results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD)
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data[0].cpu().numpy()
                
                # Definir conexões do esqueleto (pares de índices dos pontos-chave)
                skeleton = [
                    # Face
                    #(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                    # Upper body
                    #(0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    # Corpo superior
                    (5, 7), (7, 9), (6, 8), (8, 10),
                    # Tronco
                    (5, 6), (5, 11), (6, 12), (11, 12),
                    # Corpo inferior
                    (11, 13), (13, 15), (12, 14), (14, 16)
                ]
                
                # Desenhar linhas do esqueleto
                for connection in skeleton:
                    if all(keypoints[connection[0]] != 0) and all(keypoints[connection[1]] != 0):
                        pt1 = (int(keypoints[connection[0]][0]), int(keypoints[connection[0]][1]))
                        pt2 = (int(keypoints[connection[1]][0]), int(keypoints[connection[1]][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                
                # Calcular ângulos para diferentes articulações
                angles = {}
                
                # Calcular ângulo do cotovelo
                if all(keypoints[7] != 0) and all(keypoints[5] != 0) and all(keypoints[6] != 0):
                    elbow_angle = self.calculate_angle(keypoints[5], keypoints[7], keypoints[6])
                    angles['cotovelo'] = elbow_angle
                
                # Calcular ângulo do joelho (perna direita)
                if all(keypoints[11] != 0) and all(keypoints[13] != 0) and all(keypoints[12] != 0):
                    knee_angle = self.calculate_angle(keypoints[11], keypoints[13], keypoints[12])
                    angles['joelho_direito'] = knee_angle
                
                # Calcular ângulo do joelho (perna esquerda)
                if all(keypoints[10] != 0) and all(keypoints[12] != 0) and all(keypoints[11] != 0):
                    knee_angle_left = self.calculate_angle(keypoints[10], keypoints[12], keypoints[11])
                    angles['joelho_esquerdo'] = knee_angle_left
                
                # Calcular ângulo do tornozelo (perna direita)
                if all(keypoints[13] != 0) and all(keypoints[15] != 0) and all(keypoints[14] != 0):
                    ankle_angle = self.calculate_angle(keypoints[13], keypoints[15], keypoints[14])
                    angles['tornozelo_direito'] = ankle_angle
                
                # Calcular ângulo do tornozelo (perna esquerda)
                if all(keypoints[12] != 0) and all(keypoints[14] != 0) and all(keypoints[13] != 0):
                    ankle_angle_left = self.calculate_angle(keypoints[12], keypoints[14], keypoints[13])
                    angles['tornozelo_esquerdo'] = ankle_angle_left
                
                # Desenhar pontos-chave (excluindo pontos-chave do rosto)
                for i, kp in enumerate(keypoints):
                    #if all(kp != 0):
                    if all(kp != 0) and i >= 5:  # Ignorar pontos-chave do rosto (0-4)
                        cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
                
                # Desenhar ângulos no quadro
                y_offset = 30
                for joint, angle in angles.items():
                    cv2.putText(frame, f"{joint}: {angle:.1f}°", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y_offset += 30  # Mover para a próxima linha para cada ângulo
                
                # Armazenar ângulos no buffer
                if angles:
                    self.angle_buffer.append({
                        'timestamp': time.time() - self.start_time,
                        'angles': angles
                    })
                    
                    # Manter apenas as últimas medições BUFFER_SIZE
                    if len(self.angle_buffer) > config.BUFFER_SIZE:
                        self.angle_buffer.pop(0)
        
        return frame
    
    def save_data(self):
        """Salvar dados coletados em arquivos CSV e JSON."""
        if config.SAVE_DATA and self.angle_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salvar como CSV
            csv_filename = f"{config.OUTPUT_DIR}/dados_dos_movimentos_{timestamp}.csv"
            if self.angle_buffer:
                # Obter todos os nomes de articulações únicos
                joint_names = set()
                for data in self.angle_buffer:
                    joint_names.update(data['angles'].keys())
                joint_names = sorted(list(joint_names))
                
                # Escrever cabeçalho CSV
                with open(csv_filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp'] + joint_names)
                    
                    # Escrever linhas de dados
                    for data in self.angle_buffer:
                        row = [data['timestamp']]
                        for joint in joint_names:
                            row.append(data['angles'].get(joint, ''))
                        writer.writerow(row)
            
            # Criar gráfico de ângulos
            if self.angle_buffer:
                plt.figure(figsize=(10, 6))
                times = [d['timestamp'] for d in self.angle_buffer]
                
                for joint in self.angle_buffer[0]['angles'].keys():
                    angles = [d['angles'][joint] for d in self.angle_buffer]
                    plt.plot(times, angles, label=joint)
                
                plt.xlabel('Tempo (s)')
                plt.ylabel('Ângulo (graus)')
                plt.title('Ângulos das articulações ao longo do tempo')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{config.OUTPUT_DIR}/angles_plot_{timestamp}.png")
                plt.close()
    
    def run(self):
        """Loop principal para o analisador de movimento."""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Processar quadro
                processed_frame = self.process_frame(frame)
                
                # Exibir quadro
                cv2.imshow('Analisador de Movimento', processed_frame)
                # Quebrar loop com 'q' pressionado
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.save_data()
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = MovementAnalyzer()
    analyzer.run() 