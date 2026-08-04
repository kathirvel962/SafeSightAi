import os
import sys
import json
import argparse
import requests
import cv2
import time
import logging

# Add project root to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.camera import WideAngleCamera
from modules.detector import WorkerDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
DEFAULT_TEST_IMAGE = "data/test_person.jpg"

def download_sample_image(destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        logger.info(f"Sample image already exists at {destination}")
        return True
    
    logger.info(f"Downloading sample image from {SAMPLE_IMAGE_URL} to {destination}...")
    try:
        response = requests.get(SAMPLE_IMAGE_URL, timeout=10)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            logger.info("Sample image downloaded successfully.")
            return True
        else:
            logger.error(f"Failed to download image. Status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error downloading sample image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify Pretrained YOLOv8 Person Detection")
    parser.add_argument("--source", type=str, default="webcam", help="Path to image/video or 'webcam'")
    parser.add_argument("--model", type=str, default=None, help="YOLO model file override")
    parser.add_argument("--no-gui", action="store_true", help="Run in headless mode without displaying GUI")
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
            "yolo": {"model_name": "yolov8n.pt", "conf_threshold": 0.5, "person_class_id": 0}
        }

    device = config.get("device", "cpu")
    yolo_cfg = config.get("yolo", {})
    model_name = args.model if args.model else yolo_cfg.get("model_name", "yolov8n.pt")
    conf_threshold = yolo_cfg.get("conf_threshold", 0.5)
    iou_threshold = yolo_cfg.get("iou_threshold", 0.45)
    save_screenshots = yolo_cfg.get("save_screenshots", False)
    save_video = yolo_cfg.get("save_video", False)
    output_dir = yolo_cfg.get("output_dir", "output")

    # Create directories for outputs if enabled
    if save_screenshots:
        os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)
    if save_video:
        os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)

    # Initialize detector
    detector = WorkerDetector(model_name=model_name, conf_threshold=conf_threshold, iou_threshold=iou_threshold, device=device)
    if not detector.load_model():
        logger.error("Could not load YOLOv8 model. Exiting.")
        sys.exit(1)

    # Run on image
    if args.source != "webcam" and not args.source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        source_path = args.source
        if source_path == "webcam" or source_path == DEFAULT_TEST_IMAGE:
            source_path = DEFAULT_TEST_IMAGE
            if not download_sample_image(source_path):
                logger.error("Failed to obtain test image and source is set to default. Exiting.")
                sys.exit(1)

        frame = cv2.imread(source_path)
        if frame is None:
            logger.error(f"Could not read image from {source_path}")
            sys.exit(1)

        logger.info(f"Running detection on image: {source_path}")
        detections = detector.detect_workers(frame)
        
        # Draw bounding boxes and details
        for det in detections:
            bbox = det["bbox"]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            label = f"Person {det['confidence']:.2f}"
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            logger.info(f"Detected: Conf={det['confidence']:.2f}, Box={bbox}, Center={det['center']}")

        cv2.putText(frame, f"Total Detections: {len(detections)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        # Save screenshot if configured or if no-gui is requested
        if save_screenshots and len(detections) > 0:
            screenshot_path = os.path.join(output_dir, "screenshots", f"image_detection_{int(time.time())}.jpg")
            cv2.imwrite(screenshot_path, frame)
            logger.info(f"Saved detection screenshot to {screenshot_path}")

        if args.no_gui:
            output_path = "data/yolo_output.jpg"
            cv2.imwrite(output_path, frame)
            logger.info(f"Saved detection output to {output_path}")
        else:
            cv2.imshow("YOLOv8 Verification - Image", frame)
            logger.info("Press any key on the image window to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # Run on video or webcam
    else:
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
        else:
            logger.info(f"Running detection on video: {args.source}")
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                logger.error(f"Could not open video file {args.source}")
                sys.exit(1)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_target = int(cap.get(cv2.CAP_PROP_FPS))
            if fps_target <= 0:
                fps_target = 30

        # Initialize video writer if configured
        video_writer = None
        if save_video:
            video_path = os.path.join(output_dir, "videos", f"video_output_{int(time.time())}.avi")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            video_writer = cv2.VideoWriter(video_path, fourcc, fps_target, (frame_width, frame_height))
            logger.info(f"Saving processed video stream to {video_path}")

        # Setup GUI window if needed
        window_name = "YOLOv8 Verification - Stream"
        fullscreen = False
        if not args.no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        prev_time = 0
        last_screenshot_time = 0
        last_reconnect_attempt = 0
        
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
                    
                    # Create gray warning frame for display
                    import numpy as np
                    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 128
                    cv2.putText(frame, "CAMERA DISCONNECTED - RECONNECTING...", 
                                (50, frame_height // 2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Frame dimensions
            h, w = frame.shape[:2]

            detections = []
            inference_time = 0.0
            
            # Run detection only if camera is active and we obtained a valid frame
            if is_connected and not (args.source == "webcam" and not ret):
                start_time = time.time()
                detections = detector.detect_workers(frame)
                inference_time = time.time() - start_time

                # Overlay bounding boxes
                for det in detections:
                    bbox = det["bbox"]
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    label = f"Person {det['confidence']:.2f}"
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Compute FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            # Status overlays
            status_text = "CONNECTED" if is_connected else "DISCONNECTED"
            status_color = (0, 255, 0) if is_connected else (0, 0, 255)
            
            cv2.putText(frame, f"CAMERA: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Latency: {inference_time*1000:.1f}ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Workers Count: {len(detections)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Res: {w}x{h}", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'f' to toggle Fullscreen, 'q' to Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Save screenshot if configured and rate-limited to 1 per 2 seconds
            if is_connected and save_screenshots and len(detections) > 0:
                now = time.time()
                if now - last_screenshot_time > 2.0:
                    screenshot_path = os.path.join(output_dir, "screenshots", f"stream_detection_{int(now)}.jpg")
                    cv2.imwrite(screenshot_path, frame)
                    logger.info(f"Saved detection screenshot to {screenshot_path}")
                    last_screenshot_time = now

            # Write frame to video file
            if video_writer is not None:
                video_writer.write(frame)

            if args.no_gui:
                # In no-gui mode, print to stdout periodically
                if len(detections) > 0:
                    logger.info(f"Headless Stream - Camera: {status_text} | Workers: {len(detections)}")
                elif not is_connected:
                    logger.info(f"Headless Stream - Camera: {status_text} | Attempting reconnection...")
                
                # Check for termination in headless mode (simulated delay or break after 100 iterations if testing)
                time.sleep(0.03)  # limit rate in headless mode to approx 30 fps
            else:
                cv2.imshow(window_name, frame)
                
                # Key press actions
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
            
        if video_writer is not None:
            video_writer.release()
            
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
