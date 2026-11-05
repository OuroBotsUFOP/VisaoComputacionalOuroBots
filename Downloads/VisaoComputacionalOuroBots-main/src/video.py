import cv2
import numpy as np
import math
from robo import Robo

class VideoStream:
    def __init__(self, source = 0) -> None:
        self.cap = cv2.VideoCapture(source)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.heigh = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.base_timestamp = None

    def read(self) -> tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        self.timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        if self.base_timestamp is None:
            self.base_timestamp = self.timestamp

        return ret, frame

    def release(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow("Rastreamento", frame)

    def show_robo_info_frame(self, imageFrame: np.ndarray, robo: Robo, coord_F: tuple[int, int] | None) -> None:
        # Coordenadas
        print(f"Robô | Px: ({robo.x_pixel}, {robo.y_pixel}) | Real: X={robo.x_real:.1f}cm, Y={robo.y_real:.1f}cm | {robo.angulo_graus:.1f}°")
        print(f"base_timestamp: {self.base_timestamp:.2f} | timestamp: {self.timestamp:.2f} | difference {((self.timestamp - self.base_timestamp) if self.base_timestamp is not None else self.timestamp):.2f}")
        print("-" * 50)

        # Cor do centro: Branco se a leitura for real, Vermelho se o Kalman estiver prevendo
        cor_centro = (255, 255, 255) if robo.visivel else (0, 0, 255)
        cv2.circle(imageFrame, [robo.x_pixel, robo.y_pixel], 5, cor_centro, -1)
        
        # Alerta de previsão e cálculo de seta imaginária baseada no último ângulo
        if not robo.visivel:
            cv2.putText(imageFrame, "P", (robo.x_pixel - 40, robo.y_pixel - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            seta_x = int(robo.x_pixel + 30 * math.cos(robo.angulo_radianos))
            seta_y = int(robo.y_pixel + 30 * math.sin(robo.angulo_radianos))
            coord_F = (seta_x, seta_y)

        # Desenha a SETA amarela 
        if coord_F:
            cv2.arrowedLine(imageFrame, [robo.x_pixel, robo.y_pixel], coord_F, (0, 255, 255), 3, tipLength=0.2)
                    
        # Mostra o valor do ângulo, pixels e a posição REAL no canto da tela
        cv2.putText(imageFrame, f"Angulo: {robo.angulo_graus:.1f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(imageFrame, f"Px: ({robo.x_pixel}, {robo.y_pixel})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(imageFrame, f"Real: X={robo.x_real:.1f}cm Y={robo.y_real:.1f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)