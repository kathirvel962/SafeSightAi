import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


class WorkerExcavatorProximity:
    """Associate workers with excavators in normalized image space.

    Pixel distance is normalized by the frame diagonal. MiDaS is used only as
    a consistency gate because its output is relative, not metric distance.
    """

    def __init__(self, config=None):
        config = config or {}
        self.warning_distance = float(config.get("warning_distance", 0.25))
        self.danger_distance = float(config.get("danger_distance", 0.12))
        self.max_depth_difference = float(config.get("max_depth_difference", 0.25))
        if self.danger_distance >= self.warning_distance:
            raise ValueError("danger_distance must be smaller than warning_distance")

    @staticmethod
    def _bottom_center(box):
        return ((box[0] + box[2]) / 2.0, float(box[3]))

    @staticmethod
    def _closest_point(point, box):
        return (
            min(max(point[0], box[0]), box[2]),
            min(max(point[1], box[1]), box[3]),
        )

    @staticmethod
    def _depth_at_point(depth_map, point):
        if depth_map is None:
            return None
        height, width = depth_map.shape[:2]
        x = min(max(int(point[0]), 0), width - 1)
        y = min(max(int(point[1]), 0), height - 1)
        x_start, x_end = max(0, x - 3), min(width, x + 4)
        y_start, y_end = max(0, y - 3), min(height, y + 4)
        values = depth_map[y_start:y_end, x_start:x_end]
        values = values[values > 0]
        return float(np.median(values)) if values.size else None

    def evaluate(self, workers, excavators, frame_shape, depth_map=None):
        """Return each worker's closest excavator and proximity state."""
        height, width = frame_shape[:2]
        diagonal = max(math.hypot(width, height), 1.0)
        results = []

        for worker in workers:
            worker_point = self._bottom_center(worker["bbox"])
            best = None
            for excavator in excavators:
                excavator_point = self._closest_point(worker_point, excavator["bbox"])
                distance = math.hypot(
                    worker_point[0] - excavator_point[0],
                    worker_point[1] - excavator_point[1],
                ) / diagonal
                worker_depth = self._depth_at_point(depth_map, worker_point)
                excavator_depth = self._depth_at_point(depth_map, excavator_point)
                depth_difference = None
                depth_consistent = True
                if worker_depth is not None and excavator_depth is not None:
                    depth_difference = abs(worker_depth - excavator_depth) / 255.0
                    depth_consistent = depth_difference <= self.max_depth_difference
                candidate = {
                    "excavator_id": excavator["id"],
                    "normalized_distance": distance,
                    "worker_point": worker_point,
                    "excavator_point": excavator_point,
                    "worker_depth": worker_depth,
                    "excavator_depth": excavator_depth,
                    "depth_difference": depth_difference,
                    "depth_consistent": depth_consistent,
                }
                if best is None or distance < best["normalized_distance"]:
                    best = candidate

            if best is None:
                results.append({
                    "excavator_id": None,
                    "normalized_distance": None,
                    "proximity_state": "NORMAL",
                    "depth_consistent": False,
                })
                continue

            distance = best["normalized_distance"]
            if not best["depth_consistent"]:
                state = "NORMAL"
            elif distance <= self.danger_distance:
                state = "CRITICAL"
            elif distance <= self.warning_distance:
                state = "WARNING"
            else:
                state = "NORMAL"
            best["proximity_state"] = state
            results.append(best)

        return results

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
