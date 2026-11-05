import numpy as np
import math
import cv2

class Robo:
    def __init__(self, x_pixel: int = 0, y_pixel: int = 0, x_real: float = 0, y_real: float = 0, angulo_graus: float = 0) -> None:
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.x_real = x_real
        self.y_real = y_real
        self.angulo_graus = angulo_graus
        self.angulo_radianos = math.radians(self.angulo_graus)
        self.visivel = False

        # Configuração do Filtro de Kalman
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def find_pos(self, coord_F: tuple[int, int] | None, coord_D: tuple[int, int] | None, coord_E: tuple[int, int] | None, distancia_focal: float, fov_camera: int, width: float, heigh: float) -> None:

        # 1. O Kalman prevê a posição antes de saber se detectou algo
        predicao = self.kalman.predict()
        kalman_x = int(predicao[0][0])
        kalman_y = int(predicao[1][0])

        # 2. Se encontrou os 3 pontos, usa a leitura real e corrige o filtro
        if coord_F and coord_E and coord_D:
            self.visivel = True
            
            centro_x_triangulo = int((coord_F[0] + coord_D[0] + coord_E[0])/3)
            centro_y_triangulo = int((coord_F[1] + coord_D[1] + coord_E[1])/3)
            
            medicao = np.array([[np.float32(centro_x_triangulo)], [np.float32(centro_y_triangulo)]])
            self.kalman.correct(medicao)

            self.x_pixel = centro_x_triangulo
            self.y_pixel = centro_y_triangulo
                    
            dx = coord_F[0] - centro_x_triangulo
            dy = coord_F[1] - centro_y_triangulo
                    
            self.angulo_radianos = math.atan2(dy, dx)
            self.angulo_graus = math.degrees(self.angulo_radianos)

        # 3. Se perdeu a tag (piscou/sombra), assume a previsão inercial
        else:
            self.visivel = False
            self.x_pixel = kalman_x
            self.y_pixel = kalman_y
                                 
        # Extrai os valores finais em centímetros usando os pixels já consolidados (reais ou do Kalman)
        pos_x = (2 * distancia_focal * math.tan(fov_camera/2)) * (self.x_pixel / width)
        pos_y = (2 * distancia_focal * math.tan(fov_camera/2)) * (self.y_pixel / heigh)

        self.x_real = pos_x
        self.y_real = pos_y