import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WideAngleCamera:
    def __init__(self, index=0, width=640, height=480, fps=30, rotation=0, flip_h=False, flip_v=False):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.cap = None
        self.is_connected = False

    def open(self):
        logger.info(f"Opening camera {self.index} with resolution {self.width}x{self.height}...")
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera index {self.index}.")
            self.is_connected = False
            return False
        
        # Set parameters
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.is_connected = True
        return True

    def reconnect(self):
        logger.info(f"Attempting to reconnect to camera index {self.index}...")
        self.release()
        success = self.open()
        if success:
            logger.info("Camera reconnected successfully.")
            self.is_connected = True
        else:
            self.is_connected = False
        return success

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            logger.warning("Camera is not opened or active.")
            self.is_connected = False
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            self.is_connected = False
            return False, None

        # Apply transforms if requested
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.flip_h and self.flip_v:
            frame = cv2.flip(frame, -1)
        elif self.flip_h:
            frame = cv2.flip(frame, 1)
        elif self.flip_v:
            frame = cv2.flip(frame, 0)

        self.is_connected = True
        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released.")
        self.is_connected = False
