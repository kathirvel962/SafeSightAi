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
from modules.depth import DepthEstimator

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
    parser = argparse.ArgumentParser(description="Verify MiDaS Relative-Depth Estimation")
    parser.add_argument("--source", type=str, default="webcam", help="Path to image or 'webcam'")
    parser.add_argument("--model", type=str, default=None, help="MiDaS model type override (e.g. MiDaS_small)")
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
            "midas": {"model_type": "MiDaS_small"}
        }

    device = config.get("device", "cpu")
    midas_cfg = config.get("midas", {})
    model_type = args.model if args.model else midas_cfg.get("model_type", "MiDaS_small")

    # Initialize Depth Estimator
    depth_estimator = DepthEstimator(model_type=model_type, device=device)
    if not depth_estimator.load_model():
        logger.error("Could not load MiDaS model. Exiting.")
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

        logger.info(f"Running depth estimation on image: {source_path}")
        start_time = time.time()
        depth_map = depth_estimator.estimate_depth(frame)
        latency = time.time() - start_time
        
        if depth_map is not None:
            logger.info(f"Depth map generated in {latency*1000:.1f}ms. Shape: {depth_map.shape}")
            # Colorize depth map (closer is hotter/brighter)
            color_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
            
            # Stack images for display
            stacked = cv2.hconcat([frame, color_depth])
            
            if args.no_gui:
                output_path = "data/midas_output.jpg"
                cv2.imwrite(output_path, stacked)
                logger.info(f"Saved depth estimation output to {output_path}")
            else:
                cv2.imshow("MiDaS Verification - Image vs Depth", stacked)
                logger.info("MiDaS provides relative depth rather than certified distance. Brighter regions are closer.")
                logger.info("Press any key on the image window to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            logger.error("Depth estimation failed.")

    # Run on webcam
    else:
        cam_cfg = config.get("camera", {})
        camera = WideAngleCamera(
            index=cam_cfg.get("index", 0),
            width=cam_cfg.get("width", 640),
            height=cam_cfg.get("height", 480),
            fps=cam_cfg.get("fps", 30)
        )
        if not camera.open():
            logger.error("Could not open camera stream. Exiting.")
            sys.exit(1)

        logger.info("Starting live depth estimation stream...")
        logger.info("MiDaS outputs relative depth rather than certified distance. Brighter regions are closer.")
        prev_time = 0
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Failed to grab camera frame.")
                break

            start_time = time.time()
            depth_map = depth_estimator.estimate_depth(frame)
            inference_time = time.time() - start_time

            if depth_map is not None:
                # Colorize depth map
                color_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
                
                # Compute FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
                prev_time = curr_time

                # Overlay status
                cv2.putText(color_depth, f"FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(color_depth, f"Latency: {inference_time*1000:.1f}ms", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(color_depth, "Relative Depth (Inferno colormap)", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display side-by-side
                stacked = cv2.hconcat([frame, color_depth])
                cv2.imshow("MiDaS Verification - Live Stream", stacked)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
