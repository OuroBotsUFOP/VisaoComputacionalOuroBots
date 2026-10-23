import cv2
import numpy as np
from typing import Sequence

from cv2.typing import MatLike


# Encontra as mascaras para cada cor, utilizando dilate para preencher pequenas falhas
def preprocess_mascara_cores(frame: np.ndarray, mascaras: Sequence[tuple[MatLike, MatLike]]) -> list[cv2.Mat]:

    # Caixa de 1's que desliza sobre a imagem
    kernel = np.ones((5,5), "uint8")

    # blur para reduzir ruido
    blur = cv2.blur(frame, (4,4))

    hsv_frame = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    list_masks = []

    for mask in mascaras:
        mascara = cv2.inRange(hsv_frame, mask[0], mask[1])
        mascara = cv2.dilate(mascara, kernel, iterations= 1)
        list_masks.append(mascara)

    return list_masks


def preprocess_frame_cinza(frame: np.ndarray) -> np.ndarray:
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    desfoque = cv2.medianBlur(cinza, 5)

    return desfoque



