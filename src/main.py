import cv2
import numpy as np
import math

from video import VideoStream
from detector import CircleDetector
from preprocessing import preprocess_frame_cinza, preprocess_mascara_cores
from robo import Robo

def main():
    stream = VideoStream(source= 0)
    circle_detector = CircleDetector(dp= 1.3, minDist= 110, param1= 50, param2 = 80, minRadius= 40, maxRadius= 130)
    robo = Robo()

    fov_camera = 70 # graus
    distancia_focal = 14.67 # cm

    # ===============================
    # DEFINIÇÃO DAS MASCARAS DE CORES
    # ===============================
    # 1. Frente (Azul)
    frente_inf = np.array([90, 50, 80], np.uint8)
    frente_sup = np.array([130, 255, 255], np.uint8)

    # 2. Esquerda (Vermelho)
    esq_inf = np.array([160, 100, 100], np.uint8)
    esq_sup = np.array([179, 255, 255], np.uint8)

    # 3. Direita (Preto)
    dir_inf = np.array([0, 0, 0], np.uint8)
    dir_sup = np.array([179, 255, 50], np.uint8)

    lista_mascaras = [
        (frente_inf, frente_sup),
        (esq_inf, esq_sup),
        (dir_inf, dir_sup)
    ]

    while True:
        sucesso, imageFrame = stream.read()

        if not sucesso:
            print("Erro ao acessar a câmera.")
            break

        # Inicialização das variáveis de posição
        coord_F = None
        coord_E = None
        coord_D = None

        mascara_frente, mascara_esq, mascara_dir = preprocess_mascara_cores(imageFrame, lista_mascaras)
        list_masks = [mascara_frente, mascara_esq, mascara_dir]

        frame = preprocess_frame_cinza(imageFrame)

        circulos = circle_detector.detect(frame)

        if circulos is not None:
            coord_F, coord_E, coord_D = circle_detector.find_circle_pos(imageFrame, list_masks, circulos)


        # ==========================================
        # PASSO 4: CINEMÁTICA E VETOR DIRECIONAL
        # ==========================================
        # Só calcula a matemática se o computador encontrou as 3 tags do robô neste frame
        if coord_F and coord_E and coord_D:

            robo.find_pos(coord_F, coord_D, coord_E, distancia_focal, fov_camera, stream.width, stream.heigh)
                
            stream.show_robo_info_frame(imageFrame, robo, coord_F)


        # Exibe a janela
        stream.show(imageFrame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    stream.release()

if __name__ == "__main__":
    main()
        

