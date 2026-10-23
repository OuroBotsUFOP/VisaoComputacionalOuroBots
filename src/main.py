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
                       
            # Centro do triangulo formado pelos 3 circulos
            centro_x_triangulo = int((coord_F[0] + coord_D[0] + coord_E[0])/3)
            centro_y_triangulo = int((coord_F[1] + coord_D[1] + coord_E[1])/3)
            centro_robo = (centro_x_triangulo, centro_y_triangulo)
        
            # Calcula a diferença (Delta) entre o centro do robô e a etiqueta da Frente
            dx = coord_F[0] - centro_x_triangulo
            dy = coord_F[1] - centro_y_triangulo
        
            # Calcula o ângulo em graus de para onde o robô está virado
            angulo_rad = math.atan2(dy, dx)
            angulo_graus = math.degrees(angulo_rad)
                         
            # Extrai os valores finais em centímetros
            pos_x = (2 * distancia_focal * math.tan(fov_camera/2))*(centro_x_triangulo/ stream.width)
            pos_y = (2 * distancia_focal * math.tan(fov_camera/2))*(centro_y_triangulo/ stream.heigh)
               
        
            # Coordenadas
            print(f"Robô | Px: ({centro_x_triangulo}, {centro_y_triangulo}) | Real: X={pos_x:.1f}cm, Y={pos_y:.1f}cm | {angulo_graus:.1f}°")
            print(f"base_timestamp: {stream.base_timestamp:.2f} | timestamp: {stream.timestamp:.2f} | difference {((stream.timestamp - stream.base_timestamp) if stream.base_timestamp is not None else stream.timestamp):.2f}")
            print("-" * 50)

            # --- Desenho visual para confirmar que a matemática funcionou ---
            # Desenha um ponto branco no meio do robô
            cv2.circle(imageFrame, centro_robo, 5, (255, 255, 255), -1)
                    
            # Desenha a SETA amarela apontando para frente
            cv2.arrowedLine(imageFrame, centro_robo, coord_F, (0, 255, 255), 3, tipLength=0.2)
                    
            # Mostra o valor do ângulo, pixels e a posição REAL no canto da tela
            cv2.putText(imageFrame, f"Angulo: {angulo_graus:.1f} graus", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(imageFrame, f"Px: ({centro_x_triangulo}, {centro_y_triangulo})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(imageFrame, f"Real: X={pos_x:.1f}cm Y={pos_y:.1f}cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # Exibe a janela
        stream.show(imageFrame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    stream.release()

if __name__ == "__main__":
    main()
        

