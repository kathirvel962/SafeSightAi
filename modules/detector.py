import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class WorkerDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.5, iou_threshold=0.45, device="cpu"):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None

    def load_model(self):
        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA was requested but is not available. Falling back to CPU.")
                self.device = "cpu"

            logger.info(f"Loading YOLOv8 model: {self.model_name} on {self.device}...")
            self.model = YOLO(self.model_name)
            logger.info("YOLOv8 model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            return False

    def detect_workers(self, frame):
        """
        Detects people (worker class 0) in the input frame.
        Returns:
            list: A list of dicts containing bbox, confidence, center, bottom_center, dimensions.
        """
        if self.model is None:
            logger.warning("YOLO model not loaded. Call load_model() first.")
            return []

        try:
            # Perform inference. Classes=0 filters for the person class automatically.
            results = self.model.predict(
                source=frame, 
                conf=self.conf_threshold, 
                iou=self.iou_threshold,
                classes=[0], 
                device=self.device,
                verbose=False
            )
            
            detections = []
            if len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    # Bounding box coordinates in [xmin, ymin, xmax, ymax]
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    
                    xmin, ymin, xmax, ymax = xyxy
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    center_x = int(xmin + width / 2)
                    center_y = int(ymin + height / 2)
                    
                    bottom_center_x = center_x
                    bottom_center_y = int(ymax)
                    
                    detections.append({
                        "bbox": [int(xmin), int(ymin), int(xmax), int(ymax)],
                        "confidence": conf,
                        "center": (center_x, center_y),
                        "bottom_center": (bottom_center_x, bottom_center_y),
                        "dimensions": (int(width), int(height))
                    })
                    
            return detections
        except Exception as e:
            logger.error(f"Inference error in WorkerDetector: {e}")
            return []
