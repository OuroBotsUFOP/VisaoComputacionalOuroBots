import cv2
import numpy as np

from typing import Sequence

class CircleDetector:
    def __init__(self, dp: float = 1.3, minDist: int = 110, param1: int = 50, param2: int = 80, minRadius: int = 40, maxRadius: int = 130)-> None:
        self.dp = dp
        self.minDist = minDist
        self.param1 = param1
        self.param2 = param2
        self.minRadius = minRadius
        self.maxRadius = maxRadius

    def detect(self, processed_frame: np.ndarray) -> np.ndarray[tuple[int, int, int]] | None:
        circles = cv2.HoughCircles(
            processed_frame,
            cv2.HOUGH_GRADIENT,
            dp = self.dp,
            minDist = self.minDist,
            param1 = self.param1,
            param2 = self.param2,
            minRadius = self.minRadius,
            maxRadius = self.maxRadius
        )

        if circles is not None:
            return np.round(circles[0, :]).astype("int")

        return None

    def find_circle_pos(self,frame: np.ndarray, list_masks: list[cv2.Mat], circulos: np.ndarray[tuple[int, int, int]]) -> Sequence[tuple[int, int] | None]:
        coord_F = None
        coord_E = None
        coord_D = None

        for (x, y, raio) in circulos:    
            # Cria a "caixa" imaginária de 10x10 píxeis no centro do círculo encontrado
            y_min = max(0, y - 5)
            y_max = min(frame.shape[0], y + 5)
            x_min = max(0, x - 5)
            x_max = min(frame.shape[1], x + 5)

            # Recorta esse pequeno quadrado das três máscaras
            area_frente = list_masks[0][y_min:y_max, x_min:x_max]
            area_esq = list_masks[1][y_min:y_max, x_min:x_max]
            area_dir = list_masks[2][y_min:y_max, x_min:x_max]
                    
            # Testa a qual máscara este círculo pertence e SALVA a coordenada
            if cv2.countNonZero(area_frente) > 0:
                coord_F = (x, y) 
                cv2.circle(frame, (x, y), raio, (255, 0, 255), 2)
                cv2.putText(frame, "F", (x - 10, y - raio - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
            elif cv2.countNonZero(area_esq) > 0:
                coord_E = (x, y) 
                cv2.circle(frame, (x, y), raio, (255, 0, 0), 2)
                cv2.putText(frame, "E", (x - 10, y - raio - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
            elif cv2.countNonZero(area_dir) > 0:
                coord_D = (x, y)
                cv2.circle(frame, (x, y), raio, (0, 255, 0), 2)
                cv2.putText(frame, "D", (x - 10, y - raio - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return [coord_F, coord_E, coord_D]

        



