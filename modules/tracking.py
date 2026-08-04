import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

class WorkerTracker:
    def __init__(self, iou_threshold=0.3, depth_window_size=5, persistence_threshold=10):
        """
        Args:
            iou_threshold (float): Minimum IoU overlap to associate a detection with an active target.
            depth_window_size (int): Number of frames over which to calculate the rolling depth average.
            persistence_threshold (int): Maximum consecutive frames a worker can be missing before being purged.
        """
        self.iou_threshold = iou_threshold
        self.depth_window_size = depth_window_size
        self.persistence_threshold = persistence_threshold
        
        self.tracked_workers = {}  # ID -> tracked worker state dict
        self.next_id = 0
        
        logger.info(f"WorkerTracker initialized: IoU Threshold={self.iou_threshold}, "
                    f"Depth Window={self.depth_window_size}, Persistence Window={self.persistence_threshold}")

    @staticmethod
    def calculate_iou(box1, box2):
        """
        Calculates Intersection over Union (IoU) of two bounding boxes [xmin, ymin, xmax, ymax].
        """
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2

        # Determine coordinates of intersection area
        ixmin = max(xmin1, xmin2)
        iymin = max(ymin1, ymin2)
        ixmax = min(xmax1, xmax2)
        iymax = min(ymax1, ymax2)

        iw = max(0, ixmax - ixmin)
        ih = max(0, iymax - iymin)
        intersection_area = iw * ih

        # Calculate box areas
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        union_area = area1 + area2 - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def update(self, detections, depth_map, proximity_extractor):
        """
        Updates tracking list with current frame's detections and calculates smoothed depth.
        
        Args:
            detections (list): Bounding boxes returned by WorkerDetector.
            depth_map (np.ndarray): Aligned relative depth map.
            proximity_extractor (ProximityExtractor): Instance of ProximityExtractor to fetch ROI depth.
            
        Returns:
            list: List of visible workers in the current frame, sorted by nearness (closest first).
        """
        timestamp = time.time()
        
        # 1. Extract raw depth values for each incoming detection
        incoming_dets = []
        for det in detections:
            bbox = det["bbox"]
            raw_depth = proximity_extractor.extract_depth(bbox, depth_map)
            incoming_dets.append({
                "bbox": bbox,
                "confidence": det["confidence"],
                "center": det["center"],
                "bottom_center": det["bottom_center"],
                "dimensions": det["dimensions"],
                "raw_depth": raw_depth
            })

        # 2. Match active tracked workers with incoming detections
        active_track_ids = list(self.tracked_workers.keys())
        candidates = []

        for track_id in active_track_ids:
            track_box = self.tracked_workers[track_id]["bbox"]
            for det_idx, det in enumerate(incoming_dets):
                iou = self.calculate_iou(track_box, det["bbox"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, det_idx))

        # Sort candidates: highest IoU first
        candidates.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_dets = set()
        matches = []

        for iou, track_id, det_idx in candidates:
            if track_id not in matched_tracks and det_idx not in matched_dets:
                matched_tracks.add(track_id)
                matched_dets.add(det_idx)
                matches.append((track_id, det_idx))

        # 3. Process matched targets: update bbox and depth history
        for track_id, det_idx in matches:
            det = incoming_dets[det_idx]
            track_profile = self.tracked_workers[track_id]
            
            track_profile["bbox"] = det["bbox"]
            track_profile["confidence"] = det["confidence"]
            track_profile["center"] = det["center"]
            track_profile["bottom_center"] = det["bottom_center"]
            track_profile["dimensions"] = det["dimensions"]
            
            # Update rolling average history
            track_profile["depth_history"].append(det["raw_depth"])
            if len(track_profile["depth_history"]) > self.depth_window_size:
                track_profile["depth_history"].pop(0)
                
            track_profile["lost_frames"] = 0
            track_profile["visible"] = True

        # 4. Process unmatched incoming detections: spawn new trackers
        for det_idx, det in enumerate(incoming_dets):
            if det_idx not in matched_dets:
                new_worker_id = self.next_id
                self.next_id += 1
                
                self.tracked_workers[new_worker_id] = {
                    "id": new_worker_id,
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "center": det["center"],
                    "bottom_center": det["bottom_center"],
                    "dimensions": det["dimensions"],
                    "depth_history": [det["raw_depth"]],
                    "lost_frames": 0,
                    "visible": True
                }

        # 5. Process unmatched active tracked workers: increment lost frame count
        lost_track_ids = []
        for track_id in active_track_ids:
            if track_id not in matched_tracks:
                track_profile = self.tracked_workers[track_id]
                track_profile["lost_frames"] += 1
                track_profile["visible"] = False
                
                # Keep active profile history, but flag for deletion if persistence threshold exceeded
                if track_profile["lost_frames"] > self.persistence_threshold:
                    lost_track_ids.append(track_id)

        # Purge tracked profiles beyond persistence window
        for track_id in lost_track_ids:
            logger.debug(f"Worker ID {track_id} went missing for {self.persistence_threshold} frames. Purging profile.")
            del self.tracked_workers[track_id]

        # 6. Format visible workers output for the current frame
        visible_workers = []
        for track_id, w_info in self.tracked_workers.items():
            if w_info["visible"] and w_info["lost_frames"] == 0:
                smoothed_depth = float(np.mean(w_info["depth_history"]))
                visible_workers.append({
                    "id": track_id,
                    "bbox": w_info["bbox"],
                    "confidence": w_info["confidence"],
                    "center": w_info["center"],
                    "bottom_center": w_info["bottom_center"],
                    "dimensions": w_info["dimensions"],
                    "relative_depth": smoothed_depth,
                    "timestamp": timestamp
                })

        # Sort visible workers: highest depth value first (closest worker first)
        visible_workers.sort(key=lambda w: w["relative_depth"], reverse=True)
        return visible_workers
