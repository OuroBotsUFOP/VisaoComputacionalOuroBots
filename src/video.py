import cv2
import numpy as np

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