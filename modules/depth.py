import logging
import torch
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small", device="cpu"):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.transform = None

    def load_model(self):
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA was requested but is not available. Falling back to CPU.")
                self.device = "cpu"

            logger.info(f"Loading MiDaS model: {self.model_type} on {self.device}...")
            # Pre-trust the dependency repository to bypass interactive prompt
            try:
                torch.hub.list("rwightman/gen-efficientnet-pytorch", trust_repo=True)
            except Exception as e:
                logger.warning(f"Could not pre-trust rwightman repo: {e}")

            # Load model and transforms from PyTorch Hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            
            if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform

            self.model.to(self.device)
            self.model.eval()
            logger.info("MiDaS model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading MiDaS model: {e}")
            return False

    def estimate_depth(self, frame):
        """
        Generates a relative-depth map aligned with the original frame resolution.
        Returns:
            np.ndarray: Relative-depth map (normalized to 0-255 uint8 format).
        """
        if self.model is None or self.transform is None:
            logger.warning("MiDaS model not loaded. Call load_model() first.")
            return None

        try:
            # MiDaS expects RGB input
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.transform(img).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()
            
            # Normalize to 0-255 for visualization
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            
            if depth_max - depth_min > 0:
                depth_norm = (255 * (depth_map - depth_min) / (depth_max - depth_min)).astype(np.uint8)
            else:
                depth_norm = np.zeros(depth_map.shape, dtype=np.uint8)
                
            return depth_norm
        except Exception as e:
            logger.error(f"Inference error in DepthEstimator: {e}")
            return None
