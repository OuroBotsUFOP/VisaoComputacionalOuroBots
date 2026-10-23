import numpy as np
import math


class Robo:
    def __init__(self, x_pixel: int = 0, y_pixel: int = 0, x_real: float = 0, y_real: float = 0, angulo_graus: float = 0) -> None:
        self.x_pixel = x_pixel
        self.y_pixel = y_pixel
        self.x_real = x_real
        self.y_real = y_real
        self.angulo_graus = angulo_graus
        self.angulo_radianos = math.radians(self.angulo_graus)

    def find_pos(self, coord_F: tuple[int, int], coord_D: tuple[int, int], coord_E: tuple[int, int], distancia_focal: float, fov_camera: int, width: float, heigh: float) -> None:

        # Centro do triangulo formado pelos 3 circulos
        centro_x_triangulo = int((coord_F[0] + coord_D[0] + coord_E[0])/3)
        centro_y_triangulo = int((coord_F[1] + coord_D[1] + coord_E[1])/3)
        
        self.x_pixel = centro_x_triangulo
        self.y_pixel = centro_y_triangulo
                
        # Calcula a diferença (Delta) entre o centro do robô e a etiqueta da Frente
        dx = coord_F[0] - centro_x_triangulo
        dy = coord_F[1] - centro_y_triangulo
                
        # Calcula o ângulo em graus de para onde o robô está virado
        angulo_rad = math.atan2(dy, dx)
        angulo_graus = math.degrees(angulo_rad)

        self.angulo_graus = angulo_graus
        self.angulo_radianos = angulo_rad
                                 
        # Extrai os valores finais em centímetros
        pos_x = (2 * distancia_focal * math.tan(fov_camera/2))*(centro_x_triangulo/ width)
        pos_y = (2 * distancia_focal * math.tan(fov_camera/2))*(centro_y_triangulo/ heigh)

        self.x_real = pos_x
        self.y_real = pos_y