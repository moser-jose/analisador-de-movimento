import cv2
import numpy as np
from ultralytics import YOLO
import config
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
from angle_calculator import YOLOAngleCalculator, MediaPipeAngleCalculator

class MovementAnalyzer:
    def __init__(self, engine='yolo'):
        self.engine = engine.lower()
        
        # Inicializar o modelo YOLO para detecção
        if self.engine == 'yolo':
            self.model = YOLO(config.MODEL_PATH, verbose=False)
            self.angle_calculator = YOLOAngleCalculator()
        elif self.engine == 'mediapipe':
            self.model = None  # Não precisamos do modelo YOLO para MediaPipe
            self.angle_calculator = MediaPipeAngleCalculator()
            # Definir as conexões do esqueleto para MediaPipe
            self.mp_pose = self.angle_calculator.mp_pose
            self.mediapipe_connections = [
                # Braço direito
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
                (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_PINKY),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_WRIST, self.mp_pose.PoseLandmark.RIGHT_THUMB),
                (self.mp_pose.PoseLandmark.RIGHT_PINKY, self.mp_pose.PoseLandmark.RIGHT_INDEX),
                
                # Braço esquerdo
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
                (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_PINKY),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.LEFT_THUMB),
                (self.mp_pose.PoseLandmark.LEFT_PINKY, self.mp_pose.PoseLandmark.LEFT_INDEX),
                
                # Tronco
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
                
                # Perna direita
                (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
                (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_HEEL),
                (self.mp_pose.PoseLandmark.RIGHT_HEEL, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.RIGHT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),
                
                # Perna esquerda
                (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
                (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_HEEL),
                (self.mp_pose.PoseLandmark.LEFT_HEEL, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                (self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
                
                # Face (opcional, mas adiciona completude ao esqueleto)
                (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EYE_INNER),
                (self.mp_pose.PoseLandmark.LEFT_EYE_INNER, self.mp_pose.PoseLandmark.LEFT_EYE),
                (self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.LEFT_EYE_OUTER),
                (self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EYE_INNER),
                (self.mp_pose.PoseLandmark.RIGHT_EYE_INNER, self.mp_pose.PoseLandmark.RIGHT_EYE),
                (self.mp_pose.PoseLandmark.RIGHT_EYE, self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER),
                (self.mp_pose.PoseLandmark.MOUTH_LEFT, self.mp_pose.PoseLandmark.MOUTH_RIGHT),
                #(self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_EAR),
                
                # Novas conexões entre olhos e orelhas
                #(self.mp_pose.PoseLandmark.LEFT_EYE, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.RIGHT_EYE, self.mp_pose.PoseLandmark.RIGHT_EAR),
                (self.mp_pose.PoseLandmark.LEFT_EYE_OUTER, self.mp_pose.PoseLandmark.LEFT_EAR),
                (self.mp_pose.PoseLandmark.RIGHT_EYE_OUTER, self.mp_pose.PoseLandmark.RIGHT_EAR),
                #(self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.LEFT_EAR),
                #(self.mp_pose.PoseLandmark.NOSE, self.mp_pose.PoseLandmark.RIGHT_EAR),
            ]
        else:
            raise ValueError(f"Engine não reconhecido: {engine}. Use 'yolo' ou 'mediapipe'.")
        
        self.cap = cv2.VideoCapture(config.CAMERA_DEVICE)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        
        self.angle_buffer = []
        self.start_time = time.time()

    def process_frame(self, frame):
        """Processa um único quadro e detecta poses."""
        processed_frame = frame.copy()
        
        if self.engine == 'yolo':
            # Usando YOLO para detecção e cálculo de ângulos
            results = self.model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
            
            for result in results:
                if result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    
                    # Desenhar caixa delimitadora (bounding box) ao redor da pessoa
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Desenhar retângulo ao redor da pessoa
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Adicionar rótulo "Pessoa" com confiança
                        label = f"Pessoa {conf:.2f}"
                        cv2.putText(processed_frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Definir conexões do esqueleto (pares de índices dos pontos-chave)
                    skeleton = [
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
                            cv2.line(processed_frame, pt1, pt2, (0, 255, 0), 2)
                    
                    # Calcular ângulos utilizando o calculador YOLO
                    angles = self.angle_calculator.process_keypoints(keypoints)
                    
                    # Desenhar pontos-chave (excluindo pontos-chave do rosto)
                    for i, kp in enumerate(keypoints):
                        if all(kp != 0) and i >= 5:  # Ignorar pontos-chave do rosto (0-4)
                            cv2.circle(processed_frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
                    
                    # Armazenar ângulos no buffer se disponíveis
                    if angles:
                        self.angle_buffer.append({
                            'timestamp': time.time() - self.start_time,
                            'angles': angles
                        })
                        
                        # Manter apenas as últimas medições BUFFER_SIZE
                        if len(self.angle_buffer) > config.BUFFER_SIZE:
                            self.angle_buffer.pop(0)
        
        elif self.engine == 'mediapipe':
            # Usar MediaPipe para detecção e cálculo de ângulos
            angles, _, landmarks = self.angle_calculator.process_frame(frame)
            
            # Processar o frame com MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.angle_calculator.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                pose_landmarks = results.pose_landmarks.landmark
                
                # Criar caixa delimitadora ao redor da pessoa
                x_coords = []
                y_coords = []
                for landmark in pose_landmarks:
                    if landmark.visibility > 0.5:
                        x_coords.append(landmark.x * frame.shape[1])
                        y_coords.append(landmark.y * frame.shape[0])
                
                if x_coords and y_coords:
                    # Calcular as coordenadas da caixa delimitadora
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    # Adicionar algum padding à caixa
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    # Desenhar retângulo ao redor da pessoa
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Adicionar rótulo "Pessoa"
                    cv2.putText(processed_frame, "Pessoa", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Desenhar as conexões do esqueleto usando as definições personalizadas
                for connection in self.mediapipe_connections:
                    start_idx, end_idx = connection
                    
                    if (pose_landmarks[start_idx].visibility > 0.5 and
                        pose_landmarks[end_idx].visibility > 0.5):
                        
                        start_point = (
                            int(pose_landmarks[start_idx].x * frame.shape[1]),
                            int(pose_landmarks[start_idx].y * frame.shape[0])
                        )
                        
                        end_point = (
                            int(pose_landmarks[end_idx].x * frame.shape[1]),
                            int(pose_landmarks[end_idx].y * frame.shape[0])
                        )
                        
                        cv2.line(processed_frame, start_point, end_point, (0, 255, 0), 2)
                
                # Desenhar landmarks visíveis
                for i, landmark in enumerate(pose_landmarks):
                    if landmark.visibility > 0.5:
                        x = int(landmark.x * frame.shape[1])
                        y = int(landmark.y * frame.shape[0])
                        cv2.circle(processed_frame, (x, y), 4, (0, 0, 255), -1)
            
            # Armazenar ângulos no buffer se disponíveis
            if angles:
                self.angle_buffer.append({
                    'timestamp': time.time() - self.start_time,
                    'angles': angles
                })
                
                # Manter apenas as últimas medições BUFFER_SIZE
                if len(self.angle_buffer) > config.BUFFER_SIZE:
                    self.angle_buffer.pop(0)
        
        # Desenhar ângulos no quadro (independente do motor)
        y_offset = 30
        if self.angle_buffer and 'angles' in self.angle_buffer[-1]:
            angles = self.angle_buffer[-1]['angles']
            for joint, angle in angles.items():
                cv2.putText(processed_frame, f"{joint}: {angle:.1f}°", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30  # Mover para a próxima linha para cada ângulo
        
        # Mostrar qual motor está sendo usado
        cv2.putText(processed_frame, f"Motor: {self.engine.upper()}", 
                   (processed_frame.shape[1] - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return processed_frame
    
    def save_data(self):
        """Salvar dados coletados em arquivo CSV."""
        if config.SAVE_DATA and self.angle_buffer:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Salvar como CSV
            csv_filename = f"{config.OUTPUT_DIR}/dados_dos_movimentos_{self.engine}_{timestamp}.csv"
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
                
                for joint in joint_names:
                    angles = []
                    for d in self.angle_buffer:
                        if joint in d['angles']:
                            angles.append(d['angles'][joint])
                        else:
                            angles.append(None)  # Usar None para dados ausentes
                    
                    # Filtrar None antes de plotar
                    valid_times = []
                    valid_angles = []
                    for t, a in zip(times, angles):
                        if a is not None:
                            valid_times.append(t)
                            valid_angles.append(a)
                    
                    if valid_times and valid_angles:
                        plt.plot(valid_times, valid_angles, label=joint)
                
                plt.xlabel('Tempo (s)')
                plt.ylabel('Ângulo (graus)')
                plt.title(f'Ângulos das articulações ao longo do tempo - {self.engine.upper()}')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{config.OUTPUT_DIR}/angulos_articulacoes_{self.engine}_{timestamp}.png")
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
    # Configurar argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Analisador de Movimento com YOLO e MediaPipe')
    parser.add_argument('--engine', type=str, default=config.ENGINE,
                        choices=['yolo', 'mediapipe'],
                        help='Motor de análise a ser usado (yolo ou mediapipe)')
    args = parser.parse_args()
    
    # Inicializar e executar o analisador
    analyzer = MovementAnalyzer(engine=args.engine)
    analyzer.run() 