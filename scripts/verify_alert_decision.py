import os
import sys
import logging

# Add project root to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.safety import SafetyZoneManager
from modules.alert_decision import AlertDecisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_single_frame_noise():
    """
    Test that a single-frame depth noise spike does not trigger a Critical alert.
    Critical requires 3 consecutive frames.
    """
    logger.info("Running Test: Single-frame Critical noise filtering...")
    
    # Setup safety manager with standard thresholds
    safety_mgr = SafetyZoneManager({
        "warning_threshold": 97.0,
        "danger_threshold": 145.0,
        "critical_threshold": 210.0,
        "min_certainty_confidence": 0.6
    })

    alert_engine = AlertDecisionEngine({
        "critical_confirm_frames": 3,
        "danger_confirm_frames": 8,
        "warning_confirm_frames": 15,
        "hysteresis_epsilon": 10.0
    })

    worker = {"id": 0, "relative_depth": 50.0, "confidence": 0.9}

    # 1. Simulate 10 frames of SAFE depth
    for _ in range(10):
        res = alert_engine.process_worker(worker, safety_mgr)
        assert res["zone"] == "SAFE", f"Expected SAFE, got {res['zone']}"

    # 2. Simulate 2 frames of CRITICAL depth spike (220.0)
    worker["relative_depth"] = 220.0
    for i in range(2):
        res = alert_engine.process_worker(worker, safety_mgr)
        assert res["zone"] == "SAFE", f"Critical alert triggered prematurely on frame {i+1}. Expected SAFE, got {res['zone']}"

    # 3. Drop back to SAFE
    worker["relative_depth"] = 50.0
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "SAFE", f"Expected SAFE, got {res['zone']}"

    logger.info("✓ Test Passed: Noise spike successfully ignored.")

def test_warning_confirmation():
    """
    Test that Warning alerts require exactly 15 consecutive frames.
    """
    logger.info("Running Test: Warning alert temporal confirmation...")
    
    safety_mgr = SafetyZoneManager({
        "warning_threshold": 97.0,
        "danger_threshold": 145.0,
        "critical_threshold": 210.0,
        "min_certainty_confidence": 0.6
    })

    alert_engine = AlertDecisionEngine({
        "critical_confirm_frames": 3,
        "danger_confirm_frames": 8,
        "warning_confirm_frames": 15,
        "hysteresis_epsilon": 10.0
    })

    worker = {"id": 0, "relative_depth": 120.0, "confidence": 0.9} # WARNING depth

    # 1. Check first 14 frames - confirmed should remain SAFE
    for frame in range(1, 15):
        res = alert_engine.process_worker(worker, safety_mgr)
        assert res["zone"] == "SAFE", f"Warning confirmed too early on frame {frame}. Expected SAFE, got {res['zone']}"

    # 2. 15th frame - confirmed should transition to WARNING
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "WARNING", f"Warning was not confirmed on 15th frame. Expected WARNING, got {res['zone']}"

    logger.info("✓ Test Passed: Warning alerts require 15 consecutive frames.")

def test_danger_confirmation():
    """
    Test that Danger alerts require exactly 8 consecutive frames.
    """
    logger.info("Running Test: Danger alert temporal confirmation...")
    
    safety_mgr = SafetyZoneManager({
        "warning_threshold": 97.0,
        "danger_threshold": 145.0,
        "critical_threshold": 210.0,
        "min_certainty_confidence": 0.6
    })

    alert_engine = AlertDecisionEngine({
        "critical_confirm_frames": 3,
        "danger_confirm_frames": 8,
        "warning_confirm_frames": 15,
        "hysteresis_epsilon": 10.0
    })

    worker = {"id": 0, "relative_depth": 160.0, "confidence": 0.9} # DANGER depth

    # 1. Check first 7 frames - confirmed should remain SAFE
    for frame in range(1, 8):
        res = alert_engine.process_worker(worker, safety_mgr)
        assert res["zone"] == "SAFE", f"Danger confirmed too early on frame {frame}. Expected SAFE, got {res['zone']}"

    # 2. 8th frame - confirmed should transition to DANGER
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "DANGER", f"Danger was not confirmed on 8th frame. Expected DANGER, got {res['zone']}"

    logger.info("✓ Test Passed: Danger alerts require 8 consecutive frames.")

def test_hysteresis_logic():
    """
    Test hysteresis logic on the border between DANGER and CRITICAL zones.
    - Critical threshold: 210.0
    - Epsilon: 10.0
    - Critical boundary drop: below 200.0
    """
    logger.info("Running Test: Hysteresis evaluation on boundaries...")
    
    safety_mgr = SafetyZoneManager({
        "warning_threshold": 97.0,
        "danger_threshold": 145.0,
        "critical_threshold": 210.0,
        "min_certainty_confidence": 0.6
    })

    alert_engine = AlertDecisionEngine({
        "critical_confirm_frames": 3,
        "danger_confirm_frames": 8,
        "warning_confirm_frames": 15,
        "hysteresis_epsilon": 10.0
    })

    # 1. Confirm worker at CRITICAL (220.0 for 3 frames)
    worker = {"id": 0, "relative_depth": 220.0, "confidence": 0.9}
    for _ in range(3):
        res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "CRITICAL", f"Expected CRITICAL, got {res['zone']}"

    # 2. Drop depth to 205.0 (below critical thresh 210.0, but above critical - epsilon 200.0)
    worker["relative_depth"] = 205.0
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "CRITICAL", f"Hysteresis failed. Alert dropped prematurely to {res['zone']} when depth is 205.0."

    # 3. Drop depth to 195.0 (below 200.0)
    worker["relative_depth"] = 195.0
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "DANGER", f"Expected state to drop to DANGER when depth is 195.0. Got {res['zone']}"

    logger.info("✓ Test Passed: Hysteresis rules prevent alert toggling on boundaries.")

def test_immediate_alert_cancellation():
    """
    Test that moving to SAFE cancels confirmed alerts immediately without delay.
    """
    logger.info("Running Test: Immediate alert cancellation on retreat...")
    
    safety_mgr = SafetyZoneManager({
        "warning_threshold": 97.0,
        "danger_threshold": 145.0,
        "critical_threshold": 210.0,
        "min_certainty_confidence": 0.6
    })

    alert_engine = AlertDecisionEngine({
        "critical_confirm_frames": 3,
        "danger_confirm_frames": 8,
        "warning_confirm_frames": 15,
        "hysteresis_epsilon": 10.0
    })

    # 1. Confirm worker at CRITICAL
    worker = {"id": 0, "relative_depth": 220.0, "confidence": 0.9}
    for _ in range(3):
        res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "CRITICAL", f"Expected CRITICAL, got {res['zone']}"

    # 2. Move worker to SAFE depth (50.0)
    worker["relative_depth"] = 50.0
    res = alert_engine.process_worker(worker, safety_mgr)
    assert res["zone"] == "SAFE", f"Immediate cancellation failed. Expected SAFE, got {res['zone']}"

    logger.info("✓ Test Passed: Alerts cancel immediately upon moving to safe zone.")

def main():
    logger.info("=" * 60)
    logger.info("STARTING ALERT DECISION ENGINE INTEGRATION TESTS")
    logger.info("=" * 60)
    
    try:
        test_single_frame_noise()
        test_warning_confirmation()
        test_danger_confirmation()
        test_hysteresis_logic()
        test_immediate_alert_cancellation()
        
        logger.info("=" * 60)
        logger.info("ALL ALERT DECISION ENGINE TESTS PASSED SUCCESSFULLY!")
        logger.info("=" * 60)
    except AssertionError as e:
        logger.error(f"Test failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
