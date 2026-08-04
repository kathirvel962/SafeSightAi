import sys
import os
import cv2

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from modules.depth import DepthEstimator

print("Starting debug script...")
try:
    print("Initializing DepthEstimator...")
    depth_estimator = DepthEstimator(model_type="MiDaS_small", device="cpu")
    
    print("Calling load_model()...")
    res = depth_estimator.load_model()
    print("load_model() returned:", res)
    
    if res:
        print("Reading test_person.jpg...")
        frame = cv2.imread("data/test_person.jpg")
        print("Image shape:", frame.shape)
        
        print("Calling estimate_depth()...")
        depth = depth_estimator.estimate_depth(frame)
        print("estimate_depth() completed. Output shape:", depth.shape if depth is not None else None)
        
        if depth is not None:
            print("Writing output image...")
            cv2.imwrite("data/midas_output_debug.jpg", depth)
            print("Successfully saved data/midas_output_debug.jpg")
except Exception as e:
    print("EXCEPTIONAL ERROR OCCURRED:")
    import traceback
    traceback.print_exc()
