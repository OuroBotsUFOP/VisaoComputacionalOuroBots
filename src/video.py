import cv2
import numpy as np

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

    def show_robo_info_frame(self, imageFrame: np.ndarray, robo: Robo, coord_F: tuple[int, int]) -> None:
        # Coordenadas
        print(f"Robô | Px: ({robo.x_pixel}, {robo.y_pixel}) | Real: X={robo.x_real:.1f}cm, Y={robo.y_real:.1f}cm | {robo.angulo_graus:.1f}°")
        print(f"base_timestamp: {self.base_timestamp:.2f} | timestamp: {self.timestamp:.2f} | difference {((self.timestamp - self.base_timestamp) if self.base_timestamp is not None else self.timestamp):.2f}")
        print("-" * 50)

        # --- Desenho visual para confirmar que a matemática funcionou ---
        # Desenha um ponto branco no meio do robô
        cv2.circle(imageFrame, [robo.x_pixel, robo.y_pixel], 5, (255, 255, 255), -1)
                    
        # Desenha a SETA amarela apontando para frente
        cv2.arrowedLine(imageFrame, [robo.x_pixel, robo.y_pixel], coord_F, (0, 255, 255), 3, tipLength=0.2)
                    
        # Mostra o valor do ângulo, pixels e a posição REAL no canto da tela
        cv2.putText(imageFrame, f"Angulo: {robo.angulo_graus:.1f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(imageFrame, f"Px: ({robo.x_pixel}, {robo.y_pixel})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(imageFrame, f"Real: X={robo.x_real:.1f}cm Y={robo.y_real:.1f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)