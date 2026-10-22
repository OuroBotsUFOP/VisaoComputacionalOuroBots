import cv2
import numpy as np

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
