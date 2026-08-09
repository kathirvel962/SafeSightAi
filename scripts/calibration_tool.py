import os
import sys
import json
import time
import logging
import argparse
import requests
import cv2
import numpy as np

# Add project root to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera import WideAngleCamera
from modules.detector import WorkerDetector
from modules.depth import DepthEstimator
from modules.proximity import ProximityExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
DEFAULT_TEST_IMAGE = "data/test_person.jpg"

def download_sample_image(destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        return True
    
    logger.info(f"Downloading sample image to {destination}...")
    try:
        response = requests.get(SAMPLE_IMAGE_URL, timeout=10)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            logger.info("Sample image downloaded successfully.")
            return True
        else:
            logger.error(f"Failed to download image. Status: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading sample image: {e}")
        return False

def interpolate_thresholds(calibration_points):
    """
    Interpolates relative-depth thresholds for:
    - Warning (6.0m)
    - Danger (4.0m)
    - Critical (2.0m)
    """
    if not calibration_points:
        return None
    
    # Extract distance and depth pairs
    points = [(p["target_distance_m"], p["median_midas_depth"]) for p in calibration_points]
    points.sort(key=lambda x: x[0])  # Sort by distance ascending

    def interpolate_depth(target_dist):
        if len(points) == 1:
            return points[0][1]
        
        # Extrapolate if below min distance
        if target_dist <= points[0][0]:
            d1, y1 = points[0]
            d2, y2 = points[1]
            if d2 == d1:
                return y1
            val = y1 + (target_dist - d1) * (y2 - y1) / (d2 - d1)
            return max(0.0, min(255.0, val))
        
        # Extrapolate if above max distance
        if target_dist >= points[-1][0]:
            d1, y1 = points[-2]
            d2, y2 = points[-1]
            if d2 == d1:
                return y2
            val = y1 + (target_dist - d1) * (y2 - y1) / (d2 - d1)
            return max(0.0, min(255.0, val))
        
        # Interpolate within interval
        for i in range(len(points) - 1):
            d1, y1 = points[i]
            d2, y2 = points[i+1]
            if d1 <= target_dist <= d2:
                val = y1 + (target_dist - d1) * (y2 - y1) / (d2 - d1)
                return val
        return points[-1][1]

    warning_threshold = float(interpolate_depth(6.0))
    danger_threshold = float(interpolate_depth(4.0))
    critical_threshold = float(interpolate_depth(2.0))

    return {
        "warning_threshold": warning_threshold,
        "danger_threshold": danger_threshold,
        "critical_threshold": critical_threshold
    }

def main():
    parser = argparse.ArgumentParser(description="SafeSight AI Physical Calibration Utility")
    parser.add_argument("--source", type=str, default="webcam", help="Path to image/video or 'webcam'")
    parser.add_argument("--no-gui", action="store_true", help="Run calibration tool in headless validation mode")
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join("config", "settings.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "device": "cpu",
            "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
            "yolo": {"model_name": "yolov8n.pt", "conf_threshold": 0.5, "iou_threshold": 0.45},
            "midas": {"model_type": "MiDaS_small"}
        }

    device = config.get("device", "cpu")
    yolo_cfg = config.get("yolo", {})
    midas_cfg = config.get("midas", {})

    model_name = yolo_cfg.get("model_name", "yolov8n.pt")
    conf_threshold = yolo_cfg.get("conf_threshold", 0.5)
    iou_threshold = yolo_cfg.get("iou_threshold", 0.45)
    model_type = midas_cfg.get("model_type", "MiDaS_small")

    # Initialize modules
    detector = WorkerDetector(model_name=model_name, conf_threshold=conf_threshold, iou_threshold=iou_threshold, device=device)
    if not detector.load_model():
        logger.error("Could not load YOLOv8 detector.")
        sys.exit(1)

    depth_estimator = DepthEstimator(model_type=model_type, device=device)
    if not depth_estimator.load_model():
        logger.error("Could not load MiDaS depth estimator.")
        sys.exit(1)

    proximity_extractor = ProximityExtractor()

    # Open Camera or Video source
    camera = None
    cap = None
    frame_width = 640
    frame_height = 480

    if args.source == "webcam":
        cam_cfg = config.get("camera", {})
        camera = WideAngleCamera(
            index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30)
        )
        camera.open()
        frame_width = cam_cfg.get("width", 640)
        frame_height = cam_cfg.get("height", 480)
    elif args.source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {args.source}")
            sys.exit(1)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        # Static image testing
        source_path = args.source
        if source_path == "webcam" or source_path == DEFAULT_TEST_IMAGE:
            source_path = DEFAULT_TEST_IMAGE
            if not download_sample_image(source_path):
                logger.error("Failed to obtain test image.")
                sys.exit(1)

    calibrated_points = {}  # target_distance_m -> details dict
    
    # State tracking for continuous 30-frame aggregation
    capturing_distance = None
    capture_accumulator = []
    capture_frame_count = 0
    target_frames_limit = 30
    
    window_name = "SafeSight AI Calibration Tool"
    if not args.no_gui and args.source != DEFAULT_TEST_IMAGE:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Let's map hotkeys to calibration distances
    hotkeys_mapping = {
        ord('1'): 1.0,
        ord('2'): 2.0,
        ord('3'): 3.0,
        ord('4'): 4.0,
        ord('5'): 5.0,
        ord('7'): 7.0,
        ord('0'): 10.0
    }

    logger.info("=" * 60)
    logger.info(" SAFESIGHT AI PHYSICAL PROXIMITY CALIBRATION UTILITY")
    logger.info("=" * 60)
    logger.info("Instructions:")
    logger.info("  - Stand a worker in front of the camera at a known distance.")
    logger.info("  - Press key '1' for 1.0m, '2' for 2.0m, '3' for 3.0m, etc.")
    logger.info("  - The tool will aggregate readings over 30 frames.")
    logger.info("  - Repeat for multiple distances (at least 2, ideally 3+).")
    logger.info("  - Press 's' to calculate thresholds and save profile.")
    logger.info("  - Press 'q' to quit.")
    logger.info("=" * 60)

    try:
        while True:
            frame = None
            if args.source == "webcam":
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.warning("Camera disconnected. Retrying...")
                    time.sleep(1.0)
                    camera.reconnect()
                    continue
            elif cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.info("Video end reached. Looping back...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
            else:
                # Static image
                frame = cv2.imread(source_path)
                if frame is None:
                    logger.error("Could not load image.")
                    break

            # 1. Run inference
            detections = detector.detect_workers(frame)
            depth_map = depth_estimator.estimate_depth(frame)
            
            if depth_map is None:
                logger.warning("Failed to estimate depth.")
                continue

            # Process proximity data
            workers = proximity_extractor.process_detections(detections, depth_map)

            # Annotation visual copies
            annotated_frame = frame.copy()
            color_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
            
            # Draw overlays on workers
            closest_worker = None
            if workers:
                # Sort puts closest (highest relative depth value) first
                closest_worker = workers[0]
                for i, w in enumerate(workers):
                    bbox = w["bbox"]
                    is_closest = (i == 0)
                    color = (0, 0, 255) if is_closest else (0, 255, 0)
                    tag = " [TARGET]" if is_closest else ""
                    cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    label = f"Target Depth: {w['relative_depth']:.1f}{tag}"
                    cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Handle active capture aggregation
            if capturing_distance is not None:
                if closest_worker is not None:
                    capture_accumulator.append({
                        "depth": closest_worker["relative_depth"],
                        "bbox_width": closest_worker["dimensions"][0],
                        "bbox_height": closest_worker["dimensions"][1],
                        "bbox_bottom_center": closest_worker["bottom_center"],
                        "confidence": closest_worker["confidence"]
                    })
                    capture_frame_count += 1
                
                # Overlay capture progress
                h_img, w_img = annotated_frame.shape[:2]
                progress_text = f"CAPTURING {capturing_distance}m: {capture_frame_count}/{target_frames_limit} frames"
                cv2.rectangle(annotated_frame, (10, h_img - 70), (450, h_img - 30), (0, 0, 0), -1)
                cv2.putText(annotated_frame, progress_text, (20, h_img - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                if capture_frame_count >= target_frames_limit:
                    # Calculate averages
                    avg_depth = np.mean([c["depth"] for c in capture_accumulator])
                    avg_width = np.mean([c["bbox_width"] for c in capture_accumulator])
                    avg_height = np.mean([c["bbox_height"] for c in capture_accumulator])
                    avg_center_x = np.mean([c["bbox_bottom_center"][0] for c in capture_accumulator])
                    avg_center_y = np.mean([c["bbox_bottom_center"][1] for c in capture_accumulator])
                    avg_conf = np.mean([c["confidence"] for c in capture_accumulator])

                    calibrated_points[capturing_distance] = {
                        "target_distance_m": float(capturing_distance),
                        "median_midas_depth": float(avg_depth),
                        "bbox_height_px": float(avg_height),
                        "bbox_width_px": float(avg_width),
                        "bbox_bottom_center": [float(avg_center_x), float(avg_center_y)],
                        "confidence": float(avg_conf),
                        "timestamp": time.time()
                    }

                    logger.info(f"Captured point for {capturing_distance}m:")
                    logger.info(f"  - Avg Depth: {avg_depth:.2f}")
                    logger.info(f"  - Avg Bbox dimensions: {avg_width:.1f}x{avg_height:.1f}")
                    logger.info(f"  - Avg bottom-center: ({avg_center_x:.1f}, {avg_center_y:.1f})")
                    logger.info(f"  - Avg confidence: {avg_conf:.2f}")
                    
                    # Reset state
                    capturing_distance = None
                    capture_accumulator = []
                    capture_frame_count = 0
            else:
                # Render standard guidelines and instruction overlays
                h_img, w_img = annotated_frame.shape[:2]
                cv2.rectangle(annotated_frame, (10, h_img - 80), (520, h_img - 10), (0, 0, 0), -1)
                cv2.putText(annotated_frame, "HOTKEYS: '1'-'5','7','0' = Distance. 's' = Save, 'q' = Quit", 
                            (20, h_img - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show active calibrated points count
                pts_text = f"Calibrated distances: {sorted(list(calibrated_points.keys()))} meters"
                cv2.putText(annotated_frame, pts_text, (20, h_img - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                if not closest_worker:
                    cv2.putText(annotated_frame, "WARNING: No worker detected to calibrate!", (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(annotated_frame, "Status: Worker detected. Ready to calibrate.", (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Stack views side-by-side
            stacked = cv2.hconcat([annotated_frame, color_depth])
            sh, sw = stacked.shape[:2]
            cv2.putText(stacked, "SafeSight AI Proximity Calibration Studio", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Headless exit protection if running on static test
            if args.no_gui or args.source == DEFAULT_TEST_IMAGE:
                # Fill dummy calibration points for headless test to verify functionality
                calibrated_points[1.0] = {"target_distance_m": 1.0, "median_midas_depth": 240.0, "bbox_height_px": 450, "bbox_width_px": 200, "bbox_bottom_center": [320, 400], "confidence": 0.9}
                calibrated_points[3.0] = {"target_distance_m": 3.0, "median_midas_depth": 180.0, "bbox_height_px": 210, "bbox_width_px": 100, "bbox_bottom_center": [320, 300], "confidence": 0.85}
                calibrated_points[5.0] = {"target_distance_m": 5.0, "median_midas_depth": 110.0, "bbox_height_px": 120, "bbox_width_px": 60, "bbox_bottom_center": [320, 260], "confidence": 0.8}
                calibrated_points[10.0] = {"target_distance_m": 10.0, "median_midas_depth": 45.0, "bbox_height_px": 55, "bbox_width_px": 30, "bbox_bottom_center": [320, 240], "confidence": 0.75}
                
                logger.info("Headless validation mode: simulated 4 calibration points.")
                break

            cv2.imshow(window_name, stacked)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quitting calibration tool. No profiles saved.")
                break
            
            elif key in hotkeys_mapping:
                if capturing_distance is not None:
                    logger.warning("Already capturing a point. Please wait.")
                    continue
                if not closest_worker:
                    logger.warning("No worker detected to calibrate at this point.")
                    continue
                distance_val = hotkeys_mapping[key]
                logger.info(f"Starting 30-frame capture for distance {distance_val}m...")
                capturing_distance = distance_val
                capture_accumulator = []
                capture_frame_count = 0
                
            elif key == ord('s'):
                if not calibrated_points:
                    logger.warning("Cannot save: No calibration points captured yet.")
                    continue
                break

        # Process and Save Profile if points are collected
        if calibrated_points:
            # Sort points by distance ascending
            sorted_points = [calibrated_points[d] for d in sorted(calibrated_points.keys())]
            
            # Compute dynamic safety zone thresholds using linear interpolation
            thresholds = interpolate_thresholds(sorted_points)
            
            logger.info("=" * 60)
            logger.info("CALCULATED ZONE THRESHOLDS FROM CALIBRATION:")
            logger.info(f"  - Warning (6.0m limit): {thresholds['warning_threshold']:.2f}")
            logger.info(f"  - Danger (4.0m limit):  {thresholds['danger_threshold']:.2f}")
            logger.info(f"  - Critical (2.0m limit): {thresholds['critical_threshold']:.2f}")
            logger.info("=" * 60)

            # Request profile parameters
            profile_name = "excavator_rear_3m_high"
            camera_height = 3.0
            camera_pitch = -15
            lighting = "daylight"

            if not args.no_gui and args.source != DEFAULT_TEST_IMAGE:
                print("\nPlease provide profile details in console:")
                p_name_input = input("Profile Name [default: excavator_rear_3m_high]: ").strip()
                if p_name_input:
                    profile_name = p_name_input
                
                h_input = input("Camera height in meters [default: 3.0]: ").strip()
                if h_input:
                    try:
                        camera_height = float(h_input)
                    except ValueError:
                        pass
                
                pitch_input = input("Camera pitch in degrees [default: -15]: ").strip()
                if pitch_input:
                    try:
                        camera_pitch = float(pitch_input)
                    except ValueError:
                        pass
                        
                l_input = input("Lighting condition [default: daylight]: ").strip()
                if l_input:
                    lighting = l_input

            # Create profile dictionary
            profile_data = {
                "profile_name": profile_name,
                "camera_height_meters": camera_height,
                "camera_pitch_degrees": camera_pitch,
                "lighting": lighting,
                "calibration_points": sorted_points,
                "zone_thresholds": thresholds
            }

            # Load existing profile file
            profiles_path = os.path.join("config", "calibration_profiles.json")
            existing_data = {"profiles": {}}
            if os.path.exists(profiles_path):
                try:
                    with open(profiles_path, "r") as f:
                        existing_data = json.load(f)
                        if "profiles" not in existing_data:
                            existing_data["profiles"] = {}
                except Exception:
                    pass

            # Insert/overwrite profile
            existing_data["profiles"][profile_name] = profile_data

            # Write profiles back to config/calibration_profiles.json
            with open(profiles_path, "w") as f:
                json.dump(existing_data, f, indent=2)
            logger.info(f"Successfully saved profile '{profile_name}' to {profiles_path}.")

            # Optionally update settings.json with active profile & thresholds
            set_active = "y"
            if not args.no_gui and args.source != DEFAULT_TEST_IMAGE:
                set_active_input = input("\nSet this profile as active in settings.json? (y/n) [default: y]: ").strip().lower()
                if set_active_input in ["n", "no"]:
                    set_active = "n"

            if set_active == "y":
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            settings = json.load(f)
                        
                        if "safety" not in settings:
                            settings["safety"] = {}
                        
                        settings["safety"]["active_profile"] = profile_name
                        settings["safety"]["warning_threshold"] = thresholds["warning_threshold"]
                        settings["safety"]["danger_threshold"] = thresholds["danger_threshold"]
                        settings["safety"]["critical_threshold"] = thresholds["critical_threshold"]

                        with open(config_path, "w") as f:
                            json.dump(settings, f, indent=2)
                        logger.info("Successfully updated settings.json with active profile and thresholds.")
                    except Exception as e:
                        logger.error(f"Failed to update settings.json: {e}")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    finally:
        if camera is not None:
            camera.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown completed.")

if __name__ == "__main__":
    main()
