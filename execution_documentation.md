# SafeSight AI — Phase-by-Phase Execution Documentation

This document serves as a comprehensive phase-by-phase execution and verification guide for the **SafeSight AI** excavator blind-spot worker detection and proximity alert system. 

The primary objective of SafeSight AI is to build a real-time proximity-warning and operator-assistance prototype using pretrained **YOLOv8** for worker detection and **MiDaS** for monocular relative-depth estimation. **No training, fine-tuning, annotation, or dataset preparation is performed.** Instead, physical calibration maps relative depth to safety zones.

---

## System Architecture

The workflow below details how data flows through the modular system:

```mermaid
graph TD
    A[Wide-Angle USB Camera] --> B[Frame Preprocessing & Alignment]
    B --> C[YOLOv8 Person Detection]
    B --> D[MiDaS Depth Estimation]
    C --> E[Person Bounding Boxes]
    D --> F[Aligned Relative-Depth Map]
    E --> G[Worker Proximity Extraction]
    F --> G
    G --> H[Multi-Worker Tracking (ByteTrack)]
    H --> I[Safety-Zone Classification]
    I --> J[Consecutive-Frame Alert Logic]
    J --> K[Operator Monitoring Dashboard]
    J --> L[Local Hardware Alert (Buzzer/Light)]
    J --> M[Wearable Wireless Alert (ESP32)]
    J --> N[Event Logging & Near-Miss Records]
```

---

## Phase-by-Phase Execution Guide

### Phase 1: Project Setup and Independent Model Verification

#### Objective
Establish a clean, modular project structure and verify that pretrained YOLOv8 and MiDaS run independently on test media.

#### Detailed Prompt
> Create the initial SafeSight AI project for excavator blind-spot worker detection using Python, OpenCV, Ultralytics YOLOv8 and MiDaS. The system must use the original pretrained YOLOv8 model and its existing `person` class. Do not prepare, download, annotate or train any dataset. Create a clean modular project structure for camera input, worker detection, depth estimation, proximity analysis, tracking, safety-zone logic, operator alerts, wearable communication, event logging, configuration files and future edge deployment.
>
> Install and verify all required dependencies. Test pretrained YOLOv8 independently using a sample image, recorded video and webcam feed. The YOLO output must show only person detections with bounding boxes, confidence scores and the total number of people detected. Ignore all other COCO object classes.
>
> Test MiDaS independently using a sample image and webcam feed. The MiDaS output must display a relative-depth map showing nearer and farther image regions. Clearly state that MiDaS initially provides relative depth rather than certified distance in metres.
>
> Keep YOLOv8 and MiDaS as separate programs during this phase. Include proper error handling for unavailable cameras, missing weights, unsupported GPU configuration and invalid input paths. Provide clear run instructions and dependency details. Do not combine the models, define safety zones, implement alerts, add tracking or work with hardware. Stop after confirming that YOLOv8 person detection and MiDaS relative-depth estimation work independently.

#### Technical Execution Strategy
1. **Directory Structure**: Setup a clean repository containing:
   - `config/`: JSON/YAML files containing model settings, device overrides, and zone thresholds.
   - `modules/`: Distinct sub-packages for `camera`, `detector`, `depth`, `tracker`, `alerts`, `comm`, and `logger`.
   - `scripts/`: Independent diagnostic verification scripts.
2. **Environment & Dependency Setup**: Define `requirements.txt` containing `opencv-python`, `ultralytics`, `torch`, `torchvision`, and `timm`.
3. **Independent Scripts**:
   - `scripts/verify_yolo.py`: Load `yolov8n.pt`, set `classes=[0]` (person class only), run inference, and draw bounding boxes, confidence scores, and person count on the feed.
   - `scripts/verify_midas.py`: Load a lightweight MiDaS model (e.g., `MiDaS_small` or `dpt_swin2_tiny_256`), preprocess frames, run inference, normalize output, and apply a colormap (e.g., `cv2.COLORMAP_INFERNO`).
4. **Error Handling**: Graceful fallbacks for missing webcam indices, CUDA out-of-memory errors, and missing weight downloads.

#### Verification Criteria
- [ ] Requirements install cleanly via pip.
- [ ] `verify_yolo.py` runs successfully on an image, video, and webcam, displaying boxes *only* for people.
- [ ] `verify_midas.py` generates a high-contrast depth map where closer objects appear brighter (high intensity) and farther objects appear darker.

---

### Phase 2: Pretrained YOLOv8 Worker-Detection Module

#### Objective
Encapsulate YOLOv8 into a reusable, configurable detection module that extracts coordinate geometry and confidence scores for person targets.

#### Detailed Prompt
> Develop the SafeSight AI worker-detection module using only a pretrained Ultralytics YOLOv8 model. Do not use any dataset and do not perform training or fine-tuning. Configure YOLOv8 so that it processes only the pretrained `person` class and ignores every other detected object.
>
> The module must accept an image, video file or live webcam as input. For every detected person, return the bounding-box coordinates, confidence score, bounding-box centre, bottom-centre point and bounding-box dimensions. Display the person bounding box, confidence and detection count on the output frame.
>
> Add a configurable confidence threshold and non-maximum suppression settings. Use a lightweight YOLOv8 model initially because the application will later run on an edge device. Keep the model name, confidence threshold, image size and device selection in a configuration file rather than hardcoding them.
>
> Handle multiple workers independently. The output must maintain a structured detection list for every frame so that MiDaS, tracking and safety-zone modules can later consume it.
>
> Add FPS measurement, frame-processing time and camera resolution display. Save optional detection screenshots and processed videos only when enabled through configuration. Do not create labels, train the model, compare datasets, generate mAP results, integrate MiDaS or activate alerts. Stop after delivering a stable reusable worker-detection module based entirely on pretrained YOLOv8.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/detector.py` containing a `WorkerDetector` class.
2. **Parameters Mapping**: Read configuration parameters (e.g., `model_name="yolov8n.pt"`, `conf_threshold=0.5`, `iou_threshold=0.45`, `device="cuda"`) from `config/settings.json`.
3. **Structured Outputs**:
   For each frame, return a list of dictionaries:
   ```python
   {
       "bbox": [xmin, ymin, xmax, ymax],
       "confidence": float,
       "center": (x_center, y_center),
       "bottom_center": (x_center, ymax),
       "dimensions": (width, height)
   }
   ```
4. **Metadata Overlays**: Overlay FPS, inference latency, and resolution on the visualization output. Implemented toggles for frame recording.

#### Verification Criteria
- [ ] Bounding boxes, IDs (if tracked), and confidences draw correctly.
- [ ] No non-person COCO classes (e.g., cars, chairs) trigger detections.
- [ ] Module parses config files dynamically and adapts settings.

---

### Phase 3: Wide-Angle Camera Integration

#### Objective
Connect the actual wide-angle USB camera, establishing robust frame capture, orientation adjustments, and self-healing reconnection logic.

#### Detailed Prompt
> Integrate the intended wide-angle USB camera with the pretrained YOLOv8 worker-detection module. The system must continuously capture live frames using OpenCV, correct the camera orientation if necessary and pass each frame to YOLOv8 for person detection.
>
> Support configurable camera index, width, height, FPS, rotation and horizontal or vertical flipping. Detect camera disconnection and attempt safe reconnection without crashing the complete application. Show a clear system message when the camera is unavailable.
>
> Test person detection at the centre, left edge, right edge, upper region and lower region of the camera frame. Also test partially visible people and people standing behind objects that simulate the excavator boom, bucket or load. These recordings are used only for manual testing and demonstration. Do not annotate them and do not use them to train YOLOv8.
>
> Provide a full-screen monitoring view containing the live frame, person bounding boxes, confidence values, worker count, camera status and real-time FPS. Maintain the original frame for later MiDaS processing and logging.
>
> Do not perform fisheye correction unless it is explicitly enabled and verified not to reduce detection quality. Do not crop the camera frame in a way that removes blind-spot regions. Do not add MiDaS, tracking, safety zones, alarms or wearable communication. Stop after confirming stable real-time worker detection from the actual wide-angle camera.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/camera.py` containing a `WideAngleCamera` class.
2. **Reconnection Threading**: Keep frame acquisition in a dedicated thread to decouple it from processing delays. If `cap.read()` fails, release the device and continuously attempt reconnection at 2-second intervals while logging errors.
3. **Image Transforms**: Support `cv2.rotate` and `cv2.flip` based on configurations.
4. **Pipeline Pass-through**: Provide the raw original frame to downstream consumers and only perform visual bounding-box overlays on a copy of the frame.

#### Verification Criteria
- [ ] Pulling the USB camera cable outputs a "Camera Disconnected" state on screen without terminating the process.
- [ ] Plugging the cable back in automatically restores the video feed within seconds.
- [ ] Worker detection performs correctly at the frame boundaries (sides/corners) using the wide-angle camera.

---

### Phase 4: MiDaS Relative-Depth Integration

#### Objective
Merge the MiDaS depth model with the wide-angle camera pipeline, outputting synchronized relative-depth maps aligned frame-for-frame with YOLOv8 bounding boxes.

#### Detailed Prompt
> Integrate MiDaS with the existing wide-angle camera and pretrained YOLOv8 worker-detection pipeline. Both models must process the same original camera frame. YOLOv8 must detect people, while MiDaS must generate a relative-depth map for the complete frame.
>
> Keep worker detection and depth estimation as independent modules connected through a central inference pipeline. Ensure that the MiDaS output is resized and aligned exactly with the original camera frame so that YOLO bounding-box coordinates correspond correctly to the depth map.
>
> Display two views: the original camera feed containing YOLO person bounding boxes and a visual MiDaS depth map. Include the current YOLO processing time, MiDaS processing time and total pipeline FPS.
>
> Support CPU and GPU execution. Use a lightweight MiDaS model initially if the full model causes unacceptable delay. Allow YOLOv8 and MiDaS to run at different frequencies if needed, but always associate each depth map with the correct or most recent frame.
>
> Do not estimate distance in metres, classify safety zones or activate warnings in this phase. Do not train either model. The completed workflow should follow the proposed system structure of camera capture, YOLO worker detection, MiDaS relative-depth estimation, worker proximity analysis and later alert generation. Stop after stable simultaneous worker detection and depth-map generation are achieved.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/depth.py` containing a `DepthEstimator` class.
2. **Alignment & Resizing**: The raw MiDaS output is a single-channel floating-point matrix. Resize it using `cv2.resize` with `cv2.INTER_CUBIC` back to the original camera dimensions.
3. **Decoupled Frequencies**: If depth estimation is slower than worker detection, run MiDaS on a separate thread or process every N-th frame, reusing the last calculated depth map for intermediate frames, marking it with its appropriate relative latency.
4. **Visualization**: Normalize depth map values to `0-255` and apply `cv2.applyColorMap`. Display side-by-side or in stacked windows.

#### Verification Criteria
- [ ] Bounding boxes on the RGB stream correspond perfectly to the same coordinates on the colorized depth-map visualization.
- [ ] Frame latency metrics are computed independently and displayed on-screen.

---

### Phase 5: Worker-Specific Relative-Depth Extraction

#### Objective
Extract a representative, stable relative-depth value for each detected person from a localized region of interest (ROI) within their bounding box.

#### Detailed Prompt
> Develop the worker proximity-extraction module for SafeSight AI. Use the person bounding boxes returned by pretrained YOLOv8 and extract the corresponding depth values from the aligned MiDaS depth map.
>
> Do not use the average of the entire bounding box because it may contain background, ground, machinery or nearby objects. Create a configurable region inside each bounding box, preferably the central lower-middle area representing the worker’s torso and lower body. Calculate a stable median relative-depth value from valid pixels inside that region.
>
> For every detected worker, produce a structured result containing the temporary detection index, YOLO confidence, bounding-box coordinates, bounding-box width and height, centre point, bottom-centre point, MiDaS median depth value and timestamp.
>
> Support multiple workers and calculate proximity separately for each person. Sort the workers according to estimated nearness and identify the worker with the highest immediate proximity risk. Clearly label the value as relative depth rather than metres.
>
> Add protection against invalid depth pixels, bounding boxes outside the image, extremely small boxes and partially visible people. Display the relative-depth value near each person’s bounding box for debugging.
>
> Do not define Safe, Warning, Danger or Critical zones yet. Do not create physical alerts, tracking or event logs. Stop after producing reliable worker-specific relative-depth values from live camera frames.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/proximity.py` containing a `ProximityExtractor` class.
2. **Bounding Box Scaling (ROI)**: Calculate the region of interest (ROI) inside the bounding box. For box `[xmin, ymin, xmax, ymax]`, compute:
   - `x_start = xmin + 0.3 * width`, `x_end = xmax - 0.3 * width` (center-horizontal alignment).
   - `y_start = ymin + 0.4 * height`, `y_end = ymax - 0.1 * height` (lower-middle vertical alignment).
3. **Median Extraction**: Crop the depth map using the ROI slice. Calculate the statistical median (`np.median`) to filter out background noise, ground reflection, or edge pixels.
4. **Out-of-Bounds Protection**: Clip bounding boxes to image dimensions before ROI computation to prevent index errors. Discard boxes smaller than a configurable pixel size (e.g., 20x20 pixels) as noise.

#### Verification Criteria
- [ ] Walking a worker closer to the camera monotonically increases the extracted median relative-depth value.
- [ ] Bending, squatting, or carrying objects does not result in extreme depth spikes (thanks to median filtering on the lower-middle body region).

---

### Phase 6: Physical Proximity Calibration

#### Objective
Manually measure relative-depth and bounding-box variations at set physical intervals to construct a machine-specific proximity calibration profile.

#### Detailed Prompt
> Create the SafeSight AI physical calibration module without training or fine-tuning YOLOv8 or MiDaS. The purpose is to map MiDaS relative-depth behaviour and person-box characteristics to approximate proximity zones for the final camera position.
>
> Place a person at manually measured distances from the camera, such as 1, 2, 3, 4, 5, 7 and 10 metres. At each position, capture readings for different horizontal locations, including the centre, left side, right side and near the camera-frame edges.
>
> Record the measured reference distance, MiDaS median depth value, bounding-box height, bounding-box width, bounding-box bottom-centre position, YOLO confidence, camera resolution, lighting condition and timestamp. Save these readings in a calibration file.
>
> Analyse the calibration readings and create configurable thresholds for Safe, Warning, Danger and Critical zones. The system must present these results as calibrated approximate proximity or safety-zone classification, not as certified exact distance measurement. This matches the project limitation that monocular relative-depth estimation requires calibration. 
>
> Allow separate calibration profiles for different camera mounting heights, camera angles and machines. Do not use this data for YOLO training. Do not claim metric-depth accuracy without external distance-sensing hardware. Stop after producing and validating a configurable proximity-calibration profile.

#### Technical Execution Strategy
1. **Calibration Utility**: Create `scripts/calibration_tool.py`. When launched, it displays the camera view and waits for operator input. When the user specifies the physical distance (e.g., `3m`), the tool records 30 frames of raw data and calculates averages.
2. **Data Structure**: Save calibration profiles in JSON format:
   ```json
   {
     "profile_name": "excavator_rear_3m_high",
     "camera_height_meters": 3.0,
     "camera_pitch_degrees": -15,
     "calibration_points": [
       {"target_distance_m": 1.0, "median_midas_depth": 240.5, "bbox_height_px": 450},
       {"target_distance_m": 3.0, "median_midas_depth": 180.2, "bbox_height_px": 210},
       {"target_distance_m": 5.0, "median_midas_depth": 110.0, "bbox_height_px": 120},
       {"target_distance_m": 10.0, "median_midas_depth": 45.1, "bbox_height_px": 55}
     ],
     "zone_thresholds": {
       "critical_max_depth": 220, 
       "danger_max_depth": 160,
       "warning_max_depth": 90
     }
   }
   ```
3. **Threshold Calibration**: Derive zone thresholds using interpolation of relative-depth values.

#### Verification Criteria
- [ ] Calibration readings are successfully exported to `config/calibration_profiles.json`.
- [ ] Different calibration profiles can be swapped at runtime by changing configuration parameters.

---

### Phase 7: Safety-Zone Classification

#### Objective
Evaluate the proximity parameters of workers against calibration thresholds to classify them into distinct visual safety-zone levels.

#### Detailed Prompt
> Develop the SafeSight AI safety-zone classification module using the calibrated parameters from the previous phase. For every YOLO-detected worker, combine the MiDaS relative-depth value, person bounding-box size and image position to classify the worker as Safe, Warning, Danger or Critical.
>
> Keep all thresholds in an external configuration file so they can be adjusted without changing the main program. Allow each camera to have different zone settings because camera position and machine dimensions may vary.
>
> Display the assigned safety zone beside each worker. Use clear visual differences for the four zones while keeping the interface readable. Calculate the highest current risk when multiple workers are present.
>
> Add validation rules for uncertain detections. If the YOLO confidence is below the configured threshold or the depth reading is unstable, mark the result as uncertain rather than immediately assigning a critical alert.
>
> Preserve the original calibration values in logs for debugging. Do not activate a buzzer, vibration motor, relay or wearable in this phase. Do not add tracking yet. Stop after demonstrating correct live zone classification for single and multiple workers.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/safety_zones.py` containing a `SafetyZoneClassifier` class.
2. **Classification Logic**: Let $d$ be the estimated median relative depth of the worker.
   - If $d \ge \text{critical\_max\_depth}$: Classify as **Critical**.
   - If $\text{danger\_max\_depth} \le d < \text{critical\_max\_depth}$: Classify as **Danger**.
   - If $\text{warning\_max\_depth} \le d < \text{danger\_max\_depth}$: Classify as **Warning**.
   - Otherwise: Classify as **Safe**.
3. **Visual Representation**: Define a clear visual scheme for boundaries and tags:
   - **Critical**: Solid Red bounding box.
   - **Danger**: Solid Orange bounding box.
   - **Warning**: Solid Yellow bounding box.
   - **Safe**: Solid Green bounding box.
4. **Validation Logic**: Check if detection confidence is below a specific threshold (e.g., `conf < 0.4`). If so, flag the zone as `Uncertain` (Gray box) to prevent false alerts.

#### Verification Criteria
- [ ] Stepping into the physical warning zone changes the worker's overlay border color to yellow.
- [ ] Stepping into the physical critical zone updates the overlay to red and changes the state to "Critical".

---

### Phase 8: Persistent Worker Tracking

#### Objective
Incorporate multi-object tracking to assign persistent IDs to workers, smoothing depth estimation noise using temporal historical windows.

#### Detailed Prompt
> Add real-time multi-worker tracking to SafeSight AI using YOLOv8 tracking with a lightweight tracker such as ByteTrack. Continue using the original pretrained YOLOv8 person detector without dataset training or fine-tuning.
>
> Assign a persistent tracking ID to each detected person. Maintain the ID while the worker moves across consecutive frames and temporarily disappears behind an obstacle. Store the current zone, previous zone, relative-depth history, confidence history, time first detected, time last detected and highest observed risk for every worker.
>
> Ensure that multiple workers are tracked independently. Do not transfer one worker’s safety-zone history to another worker when their paths cross. Remove inactive worker records only after a configurable timeout.
>
> Smooth the MiDaS relative-depth values using a short history for each worker so that small frame-to-frame changes do not cause unstable zone classification. Keep the smoothing window configurable.
>
> Display worker IDs and current zones beside the bounding boxes. Do not activate physical alerts in this phase. Stop after providing stable worker identities and smooth zone transitions in live and recorded video.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/tracker.py` containing a `WorkerTracker` class utilizing the Ultralytics built-in ByteTrack interface (`model.track(..., persist=True)`).
2. **Worker History Data Structure**: Maintain a registry of active workers:
   ```python
   class TrackedWorker:
       def __init__(self, track_id):
           self.track_id = track_id
           self.depth_history = collections.deque(maxlen=10) # configurable smoothing window
           self.zone_history = collections.deque(maxlen=10)
           self.first_detected = time.time()
           self.last_seen = time.time()
           self.highest_risk = "Safe"
   ```
3. **Temporal Smoothing**: Compute the median depth of the historical window (`self.depth_history`) to prevent single-frame anomalies from skewing safety classification.
4. **Deregistration**: If a worker is not detected for a period greater than `inactive_timeout` (e.g., 3.0 seconds), prune their registry record.

#### Verification Criteria
- [ ] Crossing paths of two workers does not swap their assigned tracking IDs.
- [ ] Briefly stepping behind a pole or excavator bucket preserves the worker's ID and history upon reappearance.

---

### Phase 9: Consecutive-Frame Verification and Alert Decision Logic

#### Objective
Build a robust alert decision engine using temporal window verification and hysteresis rules to filter out false alerts.

#### Detailed Prompt
> Develop the SafeSight AI alert-decision engine using tracked worker information. Do not activate an alert from a single detection frame because one incorrect YOLO or MiDaS output could create a false alarm.
>
> Require a worker to remain inside Warning, Danger or Critical conditions for a configurable number of consecutive frames or minimum duration before confirming the alert. Use different confirmation requirements for each risk level, allowing Critical alerts to activate faster than Warning alerts.
>
> Add hysteresis to zone transitions. A worker must move sufficiently outside the current threshold before the system downgrades the risk. This prevents repeated switching between adjacent zones when a worker stands close to a boundary.
>
> Maintain an alert state for every tracked worker. When several workers are visible, calculate the highest system risk and prioritise the worker with the most dangerous confirmed zone. Ensure that one worker moving to safety does not stop an active alert caused by another worker.
>
> Add configurable cooldown behaviour and immediate alert cancellation when all confirmed workers move into safe conditions. Produce software-level alert events but do not yet activate physical hardware. Stop after demonstrating stable low-false-alarm alert decisions.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/alert_decision.py` containing an `AlertDecisionEngine`.
2. **Consecutive Frame Thresholds**:
   - **Critical alert confirmation**: Requires 3 consecutive frames.
   - **Danger alert confirmation**: Requires 8 consecutive frames.
   - **Warning alert confirmation**: Requires 15 consecutive frames.
3. **Hysteresis Logic**: Let $d$ be the smoothed relative depth. If the current state is *Critical*, only downgrade to *Danger* if the depth drops below $\text{critical\_max\_depth} - \epsilon$, where $\epsilon$ is a configured offset (e.g., 10 relative-depth units).
4. **Prioritization**: Loop through all active `TrackedWorker` instances, evaluate their alert states, and set the global alert index to the maximum confirmed risk level present.

#### Verification Criteria
- [ ] Passing objects that mimic a person for a brief frame or two does not trigger any safety warning.
- [ ] A worker standing directly on the boundary line between "Danger" and "Critical" does not cause the alarm to repeatedly toggle on and off.

---

### Phase 10: Operator Monitoring Dashboard

#### Objective
Develop the excavator cabin dashboard interface displaying synchronized video feeds, active alarms, system health metrics, and controls.

#### Detailed Prompt
> Build the SafeSight AI operator dashboard around the existing pretrained YOLOv8, MiDaS, tracking and safety-zone pipeline. The dashboard must provide the operator with an understandable real-time view without covering important regions of the camera feed.
>
> Display the live camera feed, person bounding boxes, worker IDs, confidence values, proximity zones, current highest risk, active alert state, worker count, FPS, camera connection, YOLO status, MiDaS status and wearable connection status.
>
> Provide controls to start monitoring, stop monitoring, pause the visual display, temporarily mute audible alerts and open recent event logs. A muted buzzer must not disable visual warnings or event recording.
>
> Show a separate compact panel for the highest-risk worker, including worker ID, current zone, time inside the zone and current relative-depth reading. Display a clear warning when the system is running with low FPS, disconnected camera or unavailable model.
>
> Keep the interface suitable for an excavator cabin display with large readable text and minimal unnecessary controls. Do not connect real machinery controls. Stop after creating a functioning operator-assistance dashboard using simulated alert outputs.

#### Technical Execution Strategy
1. **Dashboard Frame**: Develop `modules/dashboard.py` using a Python UI framework such as `Tkinter` or a browser-based dashboard powered by a local Flask/Websocket server.
2. **Visual Layout**:
   - **Primary Panel**: Original wide-angle video with worker bounding boxes and ID tracks.
   - **Secondary Panel**: Small depth-map view for verification.
   - **Status Panel**: Displays model status, camera connection, and operational FPS.
   - **Alert Indicator**: A prominent full-width banner displaying the current global state (Safe = GREEN, Warning = YELLOW, Danger = ORANGE, Critical = flashing RED).
3. **Mute Control**: Local storage boolean flag. When true, suppresses audio alarms locally but continues registering event logs and dashboard overlays.

#### Verification Criteria
- [ ] Launching the dashboard displays all active indicators and the camera stream.
- [ ] Clicking "Mute" silences the simulation indicators, but the red warning banner still flashes on critical violations.

---

### Phase 11: Local Buzzer, Light and Vibration Prototype

#### Objective
Establish local hardware connectivity using a microcontroller (ESP32/Arduino) to drive physical LED indicators, alarms, and vibration mockups.

#### Detailed Prompt
> Integrate safe prototype alert hardware with SafeSight AI. Use an ESP32, Arduino or supported GPIO controller connected to an LED, buzzer and vibration motor. The hardware must receive the confirmed alert state from the main application.
>
> Define separate physical responses for each risk level. Safe should deactivate all alarms. Warning should show a visual indicator. Danger should activate an intermittent buzzer or vibration pattern. Critical should activate continuous audible and vibration alerts until the risk is cleared.
>
> Create a reliable communication protocol between the AI application and the hardware controller. Include risk level, worker ID, timestamp and alert state. Add connection monitoring, command acknowledgement, reconnection handling and a safe fallback when communication is interrupted.
>
> Prevent repeated hardware commands from being sent every frame when the alert state has not changed. Send commands only when necessary or at a controlled heartbeat interval.
>
> Do not connect the prototype to a real excavator emergency-stop, hydraulic or engine control system. A relay may control only a demonstration lamp or model motor. Stop after demonstrating reliable local hardware responses to live safety-zone changes.

#### Technical Execution Strategy
1. **Hardware Setup**: Flash an ESP32/Arduino with serial control firmware (`firmware/local_controller.ino`). Connect:
   - Green LED to Pin A, Yellow LED to Pin B, Red LED to Pin C.
   - 5V Buzzer through a transistor circuit to PWM Pin D.
   - Vibration motor driver to Pin E.
2. **Serial Communication Protocol**: Build `modules/hardware_comm.py` communicating over USB UART. Send formatted JSON packets:
   `{"risk": "Critical", "id": 4, "timestamp": 1693829283}\n`
3. **Heartbeat Protocol**: Send a ping packet every 1.0 second. If the microcontroller misses 3 consecutive heartbeats, it must automatically enter a safe fail-soft state (deactivating active outputs and blinking a status LED).
4. **Change Event Filter**: Only transmit packets to the controller when the confirmed global risk level transitions to a new state.

#### Verification Criteria
- [ ] Stepping into a "Critical" zone activates the physical buzzer and vibration motor instantly.
- [ ] Disconnecting the serial connection to the microcontroller activates its safe diagnostic warning light.

---

### Phase 12: Worker Wearable Alert Prototype

#### Objective
Develop a wearable receiver unit providing haptic alerts to workers when a critical proximity breach occurs.

#### Detailed Prompt
> Develop the SafeSight AI worker wearable using an ESP32, vibration motor, buzzer, battery and wireless communication. The wearable must receive Warning, Danger and Critical messages from the central AI system.
>
> Implement different vibration patterns for each risk level so that the worker can understand the severity even in a noisy environment. Warning should produce a mild short vibration, Danger should produce repeated vibration and Critical should produce an urgent continuous or repeating pattern until the danger is cleared.
>
> Show wearable connection status on the operator dashboard. Add automatic reconnection, message acknowledgement, duplicate-message prevention, battery status where available and a safe notification when communication is lost.
>
> For this prototype, broadcast the alert to all connected wearables or to a manually selected wearable. Do not claim that the camera can automatically identify which wearable belongs to a particular detected person. Exact person-to-wearable association would require additional RFID, UWB, BLE positioning or identification technology.
>
> Do not connect the wearable directly to machine controls. Stop after demonstrating reliable two-way warning between the SafeSight application and the wearable device.

#### Technical Execution Strategy
1. **Wearable Firmware**: Write ESP32 firmware (`firmware/wearable_node.ino`) supporting ESP-NOW or local Wi-Fi UDP.
2. **Haptic Patterns**:
   - **Warning**: 1 short rumble (200ms ON, 1000ms OFF).
   - **Danger**: Intermittent rumble (500ms ON, 500ms OFF).
   - **Critical**: Double rumble + continuous vibration (1000ms ON, 200ms OFF).
3. **Communication Bridge**: The host PC/Jetson runs a broadcast transmitter (`modules/wearable_bridge.py`) via a local serial ESP32 bridge or local Wi-Fi socket, sending broadcast alert states.
4. **User Tracking Limitation**: Expressly label that the system broadcasts to *all* wearable nodes since individual spatial-to-ID association is not present in this configuration.

#### Verification Criteria
- [ ] Triggering a "Critical" event broadcasts a wireless signal causing the wearable unit to vibrate continuously.
- [ ] Turning off the wearable node updates the "Wearable Connection" icon on the operator dashboard to disconnected.

---

### Phase 13: Event Logging and Near-Miss Recording

#### Objective
Create a structured logging utility that records safety-critical transactions, near-miss events, and diagnostic statistics without wasting storage space.

#### Detailed Prompt
> Build the SafeSight AI event-logging module. Record only meaningful events such as worker entry, safety-zone transition, alert activation, alert escalation, alert clearance, camera disconnection, model error and wearable communication failure.
>
> Store the timestamp, camera ID, worker tracking ID, YOLO confidence, person bounding box, MiDaS relative-depth value, assigned zone, alert state, processing FPS and optional snapshot path.
>
> Avoid saving every frame because that would create excessive storage usage. Save a snapshot or short video clip only for confirmed Danger and Critical events when recording is enabled.
>
> Protect the logging system from duplicate entries caused by repeated frames. Use one event when the worker enters a zone and another when the worker leaves or escalates.
>
> Create a simple log viewer that allows events to be filtered by date, camera, risk level and worker ID. Support exporting a near-miss summary for project evaluation. Stop after completing local event logging and retrieval.

#### Technical Execution Strategy
1. **Module Creation**: Create `modules/logger.py` using Python's built-in `sqlite3` or flat JSON-Lines format.
2. **Transaction Database Schema**:
   ```sql
   CREATE TABLE IF NOT EXISTS safety_events (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       timestamp TEXT,
       camera_id TEXT,
       worker_id INTEGER,
       yolo_conf REAL,
       bbox_coords TEXT,
       midas_depth REAL,
       safety_zone TEXT,
       alert_state TEXT,
       fps REAL,
       snapshot_path TEXT
   );
   ```
3. **State Transition Filter**: Record an entry only if:
   - A worker's safety zone changes (e.g., Warning -> Danger).
   - A new worker ID is registered.
   - An alert escalates.
4. **Snapshot Storage**: When a "Danger" or "Critical" state transition is validated, crop and save the active camera frame to `logs/snapshots/<timestamp>_worker_<id>.jpg` and save the path to the database.

#### Verification Criteria
- [ ] Moving a worker back and forth between "Danger" and "Critical" creates exactly one record per transition in the database.
- [ ] Database contains valid paths to JPEG snapshots of the violations.
- [ ] Log viewer utility successfully filters records by worker ID and risk levels.

---

### Phase 14: Edge-Device Deployment and Optimisation

#### Objective
Deploy the software onto an NVIDIA Jetson edge platform, utilizing optimizations like ONNX, TensorRT, and frame skipping to meet latency goals.

#### Detailed Prompt
> Deploy the complete SafeSight AI pipeline on the selected NVIDIA Jetson edge device. The system must continue using the pretrained YOLOv8 person detector without dataset training or fine-tuning.
>
> Begin with a lightweight YOLOv8 model and an appropriate lightweight MiDaS variant. Convert or optimise the models using supported ONNX or TensorRT workflows when beneficial. Preserve person-detection reliability while reducing processing delay.
>
> Benchmark YOLO alone, MiDaS alone and the complete combined system. Record average FPS, minimum FPS, YOLO latency, MiDaS latency, total alert latency, GPU usage, CPU usage, memory consumption, device temperature and power behaviour.
>
> Allow configurable input resolution, YOLO processing frequency, MiDaS processing frequency and frame skipping. YOLO may process more frequently than MiDaS if necessary, while the latest valid depth map is reused carefully.
>
> Ensure offline operation without continuous internet. Automatically start the monitoring service when the device boots, recover from camera disconnection and save logs locally.
>
> Do not optimise only for high FPS while ignoring missed workers. Stop after producing a stable edge-AI deployment and performance report.

#### Technical Execution Strategy
1. **Model Optimization**:
   - Convert `yolov8n.pt` to ONNX using `model.export(format="onnx")` or compile to TensorRT engine (`format="engine"`) directly on the Jetson.
   - Export MiDaS to an ONNX/TensorRT format.
2. **Frequency Decoupling Logic**: Implement an execution coordinator that runs:
   - YOLOv8 inference: Every frame (e.g., 30 FPS).
   - MiDaS depth estimation: Every 3rd frame (e.g., 10 FPS).
   - Re-use the existing depth map for proximity calculations in the intermediate frames, offsetting the coordinates based on track movement vectors.
3. **Jetson System Configuration**: Configure a systemd service (`/etc/systemd/system/safesight.service`) to start the application automatically on boot.

#### Verification Criteria
- [ ] The system boots automatically and executes offline.
- [ ] Benchmarking log confirms that decoupled execution keeps the CPU/GPU temperature within bounds without dropping worker track IDs.

---

### Phase 15: Controlled System Testing

#### Objective
Validate the prototype under varying simulated operational conditions and collect structured empirical metrics.

#### Detailed Prompt
> Test the complete SafeSight AI prototype using controlled real-world scenarios. Since no annotated training or evaluation dataset is being used, create a manual test protocol for each scenario.
>
> Test scenes with no worker, one worker, multiple workers, workers entering from image edges, distant workers, workers close to the camera, partial occlusion, crouching, bending, walking, fast movement, low light, bright light, shadows, camera vibration and objects that may resemble people.
>
> For every scenario, manually record the number of visible workers, number detected by pretrained YOLOv8, missed workers, false detections, correct zone classifications, incorrect zone classifications, correct alerts, false alarms, alert delay, tracking failures and average FPS.
>
> Repeat important tests several times instead of depending on one demonstration. Give special importance to critical worker misses, danger-zone recall, false alarms per hour and end-to-end alert latency.
>
> Clearly document limitations caused by relying entirely on a general pretrained YOLOv8 model. Do not invent mAP, precision or recall values from an unlabeled dataset. Report only measurements supported by the controlled tests. Stop after generating a complete test report and improvement list.

#### Technical Execution Strategy
1. **Manual Testing Protocol**: Design a test script mapping out 15 test scenarios.
2. **Data Logging Spreadsheet**: Construct a markdown-based testing matrix in `reports/testing_report.md` tracking parameters for each run.
3. **Limitation Documentation**: Focus on detailing edge cases where pretrained YOLOv8 might fail (e.g., high-angle perspectives, mud-splattered workers wearing specialized gear, heavy occlusion by excavator buckets).

#### Verification Criteria
- [ ] Complete manual testing matrix is documented with empirical results.
- [ ] Clear latency and failure modes are explicitly detailed.

---

### Phase 16: Final Demonstration and Documentation

#### Objective
Organize all source code, design blueprints, calibration profiles, testing reports, and safety disclosures into a final project package.

#### Detailed Prompt
> Prepare the final SafeSight AI project demonstration and documentation. The final system must show wide-angle camera input, pretrained YOLOv8 person detection, MiDaS relative-depth estimation, worker-specific proximity extraction, calibrated safety-zone classification, persistent tracking, stable alert logic, operator dashboard, buzzer activation, wearable vibration and event logging.
>
> Conduct the demonstration using a controlled environment, stationary machinery, model excavator or simulated construction setup. Do not test directly around actively operating heavy machinery without certified supervision and safety controls.
>
> Document the problem statement, proposed architecture, pretrained-model approach, camera setup, MiDaS calibration method, tracking logic, safety-zone thresholds, hardware components, wearable communication, deployment process, test results, cost estimation, limitations and future improvements.
>
> Clearly state that YOLOv8 was used with pretrained weights and no dataset training or fine-tuning was performed. Explain that MiDaS provides relative depth and that the system uses calibrated approximate proximity zones rather than certified exact distance.
>
> Present SafeSight AI as a proximity-warning and operator-assistance prototype. Do not claim that it replaces certified safety equipment or independently guarantees collision prevention.
>
> Deliver the final source structure, dependency file, setup guide, system configuration, calibration profile, model files, hardware connection details, event-log format, testing report, demonstration video plan and presentation content. Stop after completing all final project materials.

#### Technical Execution Strategy
1. **Final Package Assembly**: Verify the directory layout of the workspace.
2. **ReadMe & Documentation creation**: Populate `README.md` with:
   - System Setup instructions.
   - Operating instructions.
   - Calibration guidelines.
   - Disclaimer stating: *SafeSight AI is an operator-assistance prototype and does not replace certified industrial safety systems.*
3. **Bill of Materials (BOM)**: Provide a list of recommended components (Camera, ESP32 modules, vibration motors, relays, connectors) and approximate cost.

#### Verification Criteria
- [ ] Repository is fully self-contained.
- [ ] No temporary files or datasets exist in the clean deployment package.
- [ ] Safety warnings and calibration disclaimers are prominently displayed.
