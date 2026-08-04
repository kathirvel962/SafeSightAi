import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ProximityExtractor:
    def __init__(self, min_box_size=20, roi_horizontal=(0.3, 0.7), roi_vertical=(0.4, 0.9)):
        """
        Args:
            min_box_size (int): Minimum width or height of bounding box to process.
            roi_horizontal (tuple): Horizontal crop ratio (start, end) relative to box width.
            roi_vertical (tuple): Vertical crop ratio (start, end) relative to box height.
        """
        self.min_box_size = min_box_size
        self.roi_horizontal = roi_horizontal
        self.roi_vertical = roi_vertical

    def extract_depth(self, bbox, depth_map):
        """
        Extracts the median relative-depth value from the specified ROI of the bounding box.
        """
        h_map, w_map = depth_map.shape[:2]
        xmin, ymin, xmax, ymax = bbox

        # Protection: Clip bounding box coordinates to image boundaries
        xmin = max(0, min(xmin, w_map - 1))
        xmax = max(0, min(xmax, w_map - 1))
        ymin = max(0, min(ymin, h_map - 1))
        ymax = max(0, min(ymax, h_map - 1))

        box_width = xmax - xmin
        box_height = ymax - ymin

        # Protection: Ignore extremely small boxes
        if box_width < self.min_box_size or box_height < self.min_box_size:
            logger.debug(f"Bounding box {bbox} too small (width={box_width}, height={box_height}). Skipping depth extraction.")
            return 0.0

        # Calculate ROI coordinates inside the bounding box
        roi_xmin = int(xmin + self.roi_horizontal[0] * box_width)
        roi_xmax = int(xmin + self.roi_horizontal[1] * box_width)
        roi_ymin = int(ymin + self.roi_vertical[0] * box_height)
        roi_ymax = int(ymin + self.roi_vertical[1] * box_height)

        # Enforce boundaries for ROI
        roi_xmin = max(0, min(roi_xmin, w_map - 1))
        roi_xmax = max(roi_xmin + 1, min(roi_xmax, w_map - 1))
        roi_ymin = max(0, min(roi_ymin, h_map - 1))
        roi_ymax = max(roi_ymin + 1, min(roi_ymax, h_map - 1))

        # Crop depth map slice
        depth_slice = depth_map[roi_ymin:roi_ymax, roi_xmin:roi_xmax]

        if depth_slice.size == 0:
            return 0.0

        # Protection: filter out invalid pixels (e.g. 0 values)
        valid_pixels = depth_slice[depth_slice > 0]
        
        if valid_pixels.size == 0:
            # Fall back to entire slice if all pixels are 0
            valid_pixels = depth_slice

        # Calculate stable median relative-depth value
        median_depth = float(np.median(valid_pixels))
        return median_depth

    def process_detections(self, detections, depth_map):
        """
        Extracts proximity for all workers, sorts them by nearness, and returns structured data.
        
        Args:
            detections (list): Bounding boxes returned by WorkerDetector.
            depth_map (np.ndarray): Aligned relative depth map from DepthEstimator.
            
        Returns:
            list: List of dictionaries representing workers sorted by proximity (closest first).
        """
        if depth_map is None:
            logger.warning("Depth map is empty. Cannot extract proximity.")
            return []

        timestamp = time.time()
        workers = []

        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            median_depth = self.extract_depth(bbox, depth_map)
            
            workers.append({
                "id": idx,  # temporary index for this frame
                "confidence": det["confidence"],
                "bbox": bbox,
                "dimensions": det["dimensions"],
                "center": det["center"],
                "bottom_center": det["bottom_center"],
                "relative_depth": median_depth,
                "timestamp": timestamp
            })

        # Sort workers: highest depth value first (brighter in MiDaS depth map = closer to camera)
        workers.sort(key=lambda w: w["relative_depth"], reverse=True)
        return workers
