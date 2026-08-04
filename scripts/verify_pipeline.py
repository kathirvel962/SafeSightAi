import os
import sys
import json
import argparse
import requests
import cv2
import time
import logging
import numpy as np

# Add project root to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera import WideAngleCamera

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

def main():
    parser = argparse.ArgumentParser(description="Verify Combined YOLOv8 and MiDaS Pipeline")
    parser.add_argument("--source", type=str, default="webcam", help="Path to image/video or 'webcam'")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without displaying GUI")
    parser.add_argument("--depth-freq", type=int, default=1, help="Run MiDaS every N frames")
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

    # Initialize Capture Device first if running on stream to prevent DirectShow COM deadlocks
    camera = None
    cap = None
    frame_width = 640
    frame_height = 480
    fps_target = 30

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
        fps_target = cam_cfg.get("fps", 30)
    elif args.source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        logger.info(f"Running pipeline on video file: {args.source}")
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {args.source}")
            sys.exit(1)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_target = int(cap.get(cv2.CAP_PROP_FPS))
        if fps_target <= 0:
            fps_target = 30

    # Now load detector, depth, safety, and tracking models
    from modules.detector import WorkerDetector
    from modules.depth import DepthEstimator
    from modules.proximity import ProximityExtractor
    from modules.safety import SafetyZoneManager
    from modules.tracking import WorkerTracker

    detector = WorkerDetector(model_name=model_name, conf_threshold=conf_threshold, iou_threshold=iou_threshold, device=device)
    if not detector.load_model():
        logger.error("Could not load YOLOv8 detector. Exiting.")
        sys.exit(1)

    depth_estimator = DepthEstimator(model_type=model_type, device=device)
    if not depth_estimator.load_model():
        logger.error("Could not load MiDaS depth estimator. Exiting.")
        sys.exit(1)

    proximity_extractor = ProximityExtractor()
    safety_manager = SafetyZoneManager(config.get("safety", {}))
    tracker = WorkerTracker(iou_threshold=0.3, depth_window_size=5, persistence_threshold=10)

    # Run on static image
    if args.source != "webcam" and not args.source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        source_path = args.source
        if source_path == "webcam" or source_path == DEFAULT_TEST_IMAGE:
            source_path = DEFAULT_TEST_IMAGE
            if not download_sample_image(source_path):
                logger.error("Failed to obtain test image. Exiting.")
                sys.exit(1)

        frame = cv2.imread(source_path)
        if frame is None:
            logger.error(f"Could not read image from {source_path}")
            sys.exit(1)

        logger.info("Running unified pipeline on image...")
        
        # 1. Run worker detection
        start_yolo = time.time()
        detections = detector.detect_workers(frame)
        latency_yolo = (time.time() - start_yolo) * 1000

        # 2. Run depth estimation
        start_midas = time.time()
        depth_map = depth_estimator.estimate_depth(frame)
        latency_midas = (time.time() - start_midas) * 1000

        if depth_map is None:
            logger.error("Depth estimation failed.")
            sys.exit(1)

        # Extract proximity data and track identities across frames
        workers = tracker.update(detections, depth_map, proximity_extractor)

        # Evaluate overall system threat state
        system_state, actions = safety_manager.evaluate_system_state(workers)

        # Annotate RGB frame with bounding boxes and safety zone labels
        annotated_frame = frame.copy()
        for i, worker in enumerate(workers):
            bbox = worker["bbox"]
            zone_info = safety_manager.evaluate_worker(worker)
            color = zone_info["color"]
            zone_name = zone_info["zone"]
            is_closest = (i == 0)
            tag = " [CLOSEST]" if is_closest else ""
            
            cv2.rectangle(annotated_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            label = f"ID:{worker['id']} Depth:{worker['relative_depth']:.1f} ({zone_name}){tag}"
            cv2.putText(annotated_frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            logger.info(f"Worker ID={worker['id']} | Zone={zone_name} | Relative-Depth={worker['relative_depth']:.1f} | Box={bbox} | Closest={is_closest}")

        # Apply static mock warning overlays for static visualization
        ah, aw = annotated_frame.shape[:2]
        if actions["fullscreen_flash"]:
            red_overlay = np.zeros_like(annotated_frame)
            red_overlay[:] = (0, 0, 255)
            cv2.addWeighted(annotated_frame, 0.7, red_overlay, 0.3, 0, annotated_frame)
        elif actions["border_flash"]:
            cv2.rectangle(annotated_frame, (0, 0), (aw - 1, ah - 1), (0, 0, 255), 10)

        # Print mock alerts to console
        if actions["sound_request"] is not None:
            logger.warning(f"[MOCK AUDIO ALERT] Requesting {actions['sound_request']}-frequency warning sound!")
        if actions["log_request"]:
            logger.error(f"[MOCK COLLISION EVENT] Logging Critical Proximity Risk for worker ID {actions['closest_worker_id']}!")

        # Colorize depth map (Inferno colormap)
        color_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

        # Merge views side-by-side
        stacked = cv2.hconcat([annotated_frame, color_depth])
        h, w = stacked.shape[:2]

        # Draw overall system status overlay
        status_colors = {
            "SAFE": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "DANGER": (0, 0, 255),
            "CRITICAL": (0, 0, 255)
        }
        sys_color = status_colors.get(system_state, (0, 255, 0))
        cv2.putText(stacked, f"SYSTEM STATUS: {system_state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sys_color, 2)

        # Overlay benchmarks
        cv2.putText(stacked, f"YOLO Latency: {latency_yolo:.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(stacked, f"MiDaS Latency: {latency_midas:.1f}ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(stacked, f"Workers: {len(workers)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        logger.info(f"YOLO latency: {latency_yolo:.1f}ms | MiDaS latency: {latency_midas:.1f}ms | Workers: {len(workers)} | System Status: {system_state}")

        logger.info(f"YOLO latency: {latency_yolo:.1f}ms | MiDaS latency: {latency_midas:.1f}ms | Detections: {len(detections)}")

        if args.no_gui:
            output_path = "data/pipeline_output.jpg"
            cv2.imwrite(output_path, stacked)
            logger.info(f"Saved pipeline verification output to {output_path}")
        else:
            cv2.imshow("SafeSight AI Pipeline Verification", stacked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Run on live webcam or video stream
    else:
        window_name = "SafeSight AI Pipeline - Combined View"
        fullscreen = False
        if not args.no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = 0
        last_reconnect_attempt = 0
        last_sound_log_time = 0.0
        last_collision_log_time = 0.0
        
        frame_idx = 0
        cached_depth_map = None
        
        latency_yolo = 0.0
        latency_midas = 0.0

        while True:
            frame = None
            is_connected = False

            if args.source == "webcam":
                ret, frame = camera.read()
                is_connected = camera.is_connected
            else:
                ret, frame = cap.read()
                is_connected = ret

            if not ret or frame is None:
                if args.source != "webcam":
                    logger.info("End of video file reached.")
                    break
                else:
                    # Attempt reconnection
                    now = time.time()
                    if now - last_reconnect_attempt > 2.0:
                        logger.warning("Webcam feed lost. Attempting auto-reconnection...")
                        camera.reconnect()
                        last_reconnect_attempt = now
                    
                    # Create dummy warning frame
                    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 128
                    cv2.putText(frame, "CAMERA DISCONNECTED - RECONNECTING...", 
                                (50, frame_height // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Create black depth map matching size
                    cached_depth_map = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Define variables
            h, w = frame.shape[:2]
            workers = []

            if is_connected and not (args.source == "webcam" and not ret):
                # 1. Run worker detection on every frame
                start_yolo = time.time()
                detections = detector.detect_workers(frame)
                latency_yolo = (time.time() - start_yolo) * 1000

                # 2. Run depth estimation at configured frequency
                if frame_idx % args.depth_freq == 0 or cached_depth_map is None:
                    start_midas = time.time()
                    depth = depth_estimator.estimate_depth(frame)
                    latency_midas = (time.time() - start_midas) * 1000
                    if depth is not None:
                        cached_depth_map = depth
                
                # Extract proximity data and track identities across frames
                if cached_depth_map is not None:
                    workers = tracker.update(detections, cached_depth_map, proximity_extractor)
                
                # Evaluate overall system threat state
                system_state, actions = safety_manager.evaluate_system_state(workers)

                # Annotate RGB frame with bounding boxes and safety zone labels
                for i, worker in enumerate(workers):
                    bbox = worker["bbox"]
                    zone_info = safety_manager.evaluate_worker(worker)
                    color = zone_info["color"]
                    zone_name = zone_info["zone"]
                    is_closest = (i == 0)
                    tag = " [CLOSEST]" if is_closest else ""
                    
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    label = f"ID:{worker['id']} Depth:{worker['relative_depth']:.1f} ({zone_name}){tag}"
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Apply dynamic flashing overlays
                flash_on = int(time.time() * 4) % 2 == 0
                if actions["fullscreen_flash"] and flash_on:
                    red_overlay = np.zeros_like(frame)
                    red_overlay[:] = (0, 0, 255)
                    cv2.addWeighted(frame, 0.75, red_overlay, 0.25, 0, frame)
                elif actions["border_flash"] and flash_on:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

                # Print rate-limited mock alerts to console
                now = time.time()
                if actions["sound_request"] is not None and now - last_sound_log_time > 2.0:
                    logger.warning(f"[MOCK AUDIO ALERT] Requesting {actions['sound_request']}-frequency warning sound!")
                    last_sound_log_time = now
                if actions["log_request"] and now - last_collision_log_time > 2.0:
                    logger.error(f"[MOCK COLLISION EVENT] Logging Critical Proximity Risk for worker ID {actions['closest_worker_id']}!")
                    last_collision_log_time = now
            else:
                latency_yolo = 0.0
                latency_midas = 0.0
                system_state = "SAFE"
                actions = {"sound_request": None, "log_request": False}

            # Colorize the cached/current depth map
            if cached_depth_map is not None:
                color_depth = cv2.applyColorMap(cached_depth_map, cv2.COLORMAP_INFERNO)
            else:
                color_depth = np.zeros((h, w, 3), dtype=np.uint8)

            # Merge RGB and Depth views side-by-side
            stacked = cv2.hconcat([frame, color_depth])
            sh, sw = stacked.shape[:2]

            # Compute combined pipeline FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Status overlays
            status_text = "CONNECTED" if is_connected else "DISCONNECTED"
            status_color = (0, 255, 0) if is_connected else (0, 0, 255)
            
            # System Status Banner
            status_colors = {
                "SAFE": (0, 255, 0),
                "WARNING": (0, 255, 255),
                "DANGER": (0, 0, 255),
                "CRITICAL": (0, 0, 255)
            }
            sys_color = status_colors.get(system_state, (0, 255, 0))
            cv2.putText(stacked, f"SYSTEM STATUS: {system_state}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, sys_color, 2)
            
            cv2.putText(stacked, f"CAMERA: {status_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(stacked, f"Pipeline FPS: {fps:.1f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stacked, f"YOLO Latency: {latency_yolo:.1f}ms", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stacked, f"MiDaS Latency: {latency_midas:.1f}ms", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stacked, f"Workers: {len(workers)}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stacked, f"Decoupling Rate: {args.depth_freq}x", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(stacked, "Press 'f' to toggle Fullscreen, 'q' to Quit", (20, sh - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            frame_idx += 1

            if args.no_gui:
                # Headless logging
                if is_connected:
                    closest_text = f" | Closest Worker: ID={workers[0]['id']} Depth={workers[0]['relative_depth']:.1f} ({system_state})" if workers else ""
                    logger.info(f"Pipeline Running - Workers: {len(workers)} | Status: {system_state} | FPS: {fps:.1f} | YOLO: {latency_yolo:.1f}ms | MiDaS: {latency_midas:.1f}ms{closest_text}")
                time.sleep(0.01)
            else:
                cv2.imshow(window_name, stacked)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if args.source == "webcam":
            camera.release()
        else:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
