import numpy as np
import mediapipe as mp
import cv2

class AngleCalculator:
    """Classe base para calculadores de ângulos."""
    def calculate_angle(self, p1, p2, p3):
        """Calcula o ângulo entre três pontos."""
        a = np.array(p1) - np.array(p2)
        b = np.array(p3) - np.array(p2)
        
        cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        # Lidar com erros de precisão numérica
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return np.degrees(angle)    

class YOLOAngleCalculator(AngleCalculator):
    """Calculador de ângulos usando pontos-chave do YOLO."""
    def __init__(self):
        super().__init__()
        # Mapeamento das articulações YOLO
        self.joint_map = {
            'cotovelo_direito': [5, 7, 9],  # ombro, cotovelo, pulso
            'cotovelo_esquerdo': [6, 8, 10],
            'ombro_direito': [11, 5, 7],  # quadril, ombro, cotovelo
            'ombro_esquerdo': [12, 6, 8],
            'quadril_direito': [5, 11, 13],  # ombro, quadril, joelho
            'quadril_esquerdo': [6, 12, 14],
            'joelho_direito': [11, 13, 15],  # quadril, joelho, tornozelo
            'joelho_esquerdo': [12, 14, 16],
            'tornozelo_direito': [13, 15, 0],  # joelho, tornozelo, ponta do pé
            'tornozelo_esquerdo': [14, 16, 0]
        }
    
    def process_keypoints(self, keypoints):
        """Processa os pontos-chave do YOLO e calcula os ângulos."""
        angles = {}
        
        for joint_name, indices in self.joint_map.items():
            # Pular tornozelos se não tivermos pontos de pé
            if joint_name.startswith('tornozelo') and indices[2] == 0:
                continue
                
            p1_idx, p2_idx, p3_idx = indices
            
            # Verificar se os pontos-chave necessários estão disponíveis
            if (all(keypoints[p1_idx] != 0) and 
                all(keypoints[p2_idx] != 0) and 
                (p3_idx == 0 or all(keypoints[p3_idx] != 0))):
                
                # Se não temos o terceiro ponto para tornozelo, estimar um ponto abaixo
                if p3_idx == 0 and joint_name.startswith('tornozelo'):
                    # Estimar posição do pé abaixo do tornozelo
                    ankle = keypoints[p2_idx]
                    estimated_foot = [ankle[0], ankle[1] + 30, ankle[2]]  # 30 pixels abaixo
                    angle = self.calculate_angle(keypoints[p1_idx], keypoints[p2_idx], estimated_foot)
                else:
                    angle = self.calculate_angle(keypoints[p1_idx], keypoints[p2_idx], keypoints[p3_idx])
                
                angles[joint_name] = angle
        
        return angles

class MediaPipeAngleCalculator(AngleCalculator):
    """Calculador de ângulos usando pontos-chave do MediaPipe."""
    def __init__(self):
        super().__init__()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # Mapeamento das articulações MediaPipe
        self.joint_map = {
            'cotovelo_direito': [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                self.mp_pose.PoseLandmark.RIGHT_ELBOW, 
                                self.mp_pose.PoseLandmark.RIGHT_WRIST],
            'cotovelo_esquerdo': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                self.mp_pose.PoseLandmark.LEFT_ELBOW, 
                                self.mp_pose.PoseLandmark.LEFT_WRIST],
            'ombro_direito': [self.mp_pose.PoseLandmark.RIGHT_HIP, 
                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                            self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            'ombro_esquerdo': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                            self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                            self.mp_pose.PoseLandmark.LEFT_ELBOW],
            'quadril_direito': [self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                self.mp_pose.PoseLandmark.RIGHT_HIP, 
                                self.mp_pose.PoseLandmark.RIGHT_KNEE],
            'quadril_esquerdo': [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                self.mp_pose.PoseLandmark.LEFT_HIP, 
                                self.mp_pose.PoseLandmark.LEFT_KNEE],
            'joelho_direito': [self.mp_pose.PoseLandmark.RIGHT_HIP, 
                            self.mp_pose.PoseLandmark.RIGHT_KNEE, 
                            self.mp_pose.PoseLandmark.RIGHT_ANKLE],
            'joelho_esquerdo': [self.mp_pose.PoseLandmark.LEFT_HIP, 
                            self.mp_pose.PoseLandmark.LEFT_KNEE, 
                            self.mp_pose.PoseLandmark.LEFT_ANKLE],
            'tornozelo_direito': [self.mp_pose.PoseLandmark.RIGHT_KNEE, 
                                self.mp_pose.PoseLandmark.RIGHT_ANKLE, 
                                self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX],
            'tornozelo_esquerdo': [self.mp_pose.PoseLandmark.LEFT_KNEE, 
                                self.mp_pose.PoseLandmark.LEFT_ANKLE, 
                                self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        }

    def process_frame(self, frame):
        """Processa um frame com MediaPipe e calcula os ângulos."""
        # Converter BGR para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        angles = {}
        skeleton_connections = []
        visible_landmarks = []
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Calcular ângulos para cada articulação definida
            for joint_name, indices in self.joint_map.items():
                p1_idx, p2_idx, p3_idx = indices
                
                # Verificar se os landmarks necessários são visíveis
                if (landmarks[p1_idx].visibility > 0.5 and 
                    landmarks[p2_idx].visibility > 0.5 and 
                    landmarks[p3_idx].visibility > 0.5):
                    
                    p1 = [landmarks[p1_idx].x * frame.shape[1], 
                          landmarks[p1_idx].y * frame.shape[0]]
                    p2 = [landmarks[p2_idx].x * frame.shape[1], 
                          landmarks[p2_idx].y * frame.shape[0]]
                    p3 = [landmarks[p3_idx].x * frame.shape[1], 
                          landmarks[p3_idx].y * frame.shape[0]]
                    
                    angle = self.calculate_angle(p1, p2, p3)
                    angles[joint_name] = angle
                    
                    # Adicionar conexões para desenhar o esqueleto
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    pt3 = (int(p3[0]), int(p3[1]))
                    
                    skeleton_connections.append((pt1, pt2))
                    skeleton_connections.append((pt2, pt3))
            
            # Obter coordenadas de todos os landmarks visíveis para desenhar
            visible_landmarks = []
            for i, landmark in enumerate(landmarks):
                if landmark.visibility > 0.5:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    visible_landmarks.append((x, y, i))
        
        return angles, skeleton_connections, visible_landmarks

# Importar apenas se este arquivo for executado diretamente
if __name__ == "__main__":
    # Exemplo de código para testar os calculadores
    mp_calc = MediaPipeAngleCalculator()
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        angles, connections, landmarks = mp_calc.process_frame(frame)
        
        # Desenhar esqueleto
        for conn in connections:
            cv2.line(frame, conn[0], conn[1], (0, 255, 0), 2)
            
        # Desenhar landmarks
        for lm in landmarks:
            cv2.circle(frame, (lm[0], lm[1]), 4, (0, 0, 255), -1)
            
        # Mostrar ângulos
        y_offset = 30
        for joint, angle in angles.items():
            cv2.putText(frame, f"{joint}: {angle:.1f}°", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
            
        cv2.imshow('MediaPipe Angle Calculator Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows() 