import logging

logger = logging.getLogger(__name__)

class AlertDecisionEngine:
    def __init__(self, config=None):
        """
        Args:
            config (dict): Configurations containing:
                - critical_confirm_frames (int): default 3
                - danger_confirm_frames (int): default 8
                - warning_confirm_frames (int): default 15
                - hysteresis_epsilon (float): default 10.0
        """
        if config is None:
            config = {}

        self.critical_confirm_frames = config.get("critical_confirm_frames", 3)
        self.danger_confirm_frames = config.get("danger_confirm_frames", 8)
        self.warning_confirm_frames = config.get("warning_confirm_frames", 15)
        self.hysteresis_epsilon = config.get("hysteresis_epsilon", 10.0)

        # Map tracker worker ID -> alert state dict
        self.states = {}

        logger.info(f"AlertDecisionEngine initialized: "
                    f"CriticalConfirm={self.critical_confirm_frames} frames, "
                    f"DangerConfirm={self.danger_confirm_frames} frames, "
                    f"WarningConfirm={self.warning_confirm_frames} frames, "
                    f"HysteresisEpsilon={self.hysteresis_epsilon}")

    def prune_inactive_workers(self, active_ids):
        """
        Removes saved state for workers that are no longer active in the tracking registry.
        """
        inactive_ids = [wid for wid in self.states if wid not in active_ids]
        for wid in inactive_ids:
            logger.debug(f"Worker ID {wid} is inactive. Pruning alert state.")
            del self.states[wid]

    def process_worker(self, worker, safety_manager):
        """
        Processes a single worker, computes their confirmed alert state using temporal 
        windows and hysteresis, and returns the updated confirmed zone.
        
        Args:
            worker (dict): Tracked worker dictionary.
            safety_manager (SafetyZoneManager): Configured safety manager containing thresholds.
            
        Returns:
            dict: Zone info containing confirmed status, color, sound, and log requests.
        """
        worker_id = worker["id"]
        depth = worker.get("relative_depth", 0.0)
        confidence = worker.get("confidence", 1.0)

        # Initialize tracking states if new worker
        if worker_id not in self.states:
            self.states[worker_id] = {
                "confirmed_state": "SAFE",
                "consecutive_critical": 0,
                "consecutive_danger": 0,
                "consecutive_warning": 0
            }

        w_state = self.states[worker_id]
        current_confirmed = w_state["confirmed_state"]

        # Fetch thresholds
        warning_thresh = safety_manager.warning_threshold
        danger_thresh = safety_manager.danger_threshold
        critical_thresh = safety_manager.critical_threshold
        epsilon = self.hysteresis_epsilon

        # 1. Determine raw target zone taking uncertainty and hysteresis into account
        if confidence < safety_manager.min_certainty_confidence or depth <= 0.0:
            raw_zone = "UNCERTAIN"
        else:
            if current_confirmed == "CRITICAL":
                if depth < critical_thresh - epsilon:
                    if depth < danger_thresh - epsilon:
                        if depth < warning_thresh - epsilon:
                            raw_zone = "SAFE"
                        else:
                            raw_zone = "WARNING"
                    else:
                        raw_zone = "DANGER"
                else:
                    raw_zone = "CRITICAL"
            elif current_confirmed == "DANGER":
                if depth >= critical_thresh:
                    raw_zone = "CRITICAL"
                elif depth < danger_thresh - epsilon:
                    if depth < warning_thresh - epsilon:
                        raw_zone = "SAFE"
                    else:
                        raw_zone = "WARNING"
                else:
                    raw_zone = "DANGER"
            elif current_confirmed == "WARNING":
                if depth >= critical_thresh:
                    raw_zone = "CRITICAL"
                elif depth >= danger_thresh:
                    raw_zone = "DANGER"
                elif depth < warning_thresh - epsilon:
                    raw_zone = "SAFE"
                else:
                    raw_zone = "WARNING"
            else:  # SAFE or UNCERTAIN
                if depth >= critical_thresh:
                    raw_zone = "CRITICAL"
                elif depth >= danger_thresh:
                    raw_zone = "DANGER"
                elif depth >= warning_thresh:
                    raw_zone = "WARNING"
                else:
                    raw_zone = "SAFE"

        # Priority Rank: SAFE (0) < UNCERTAIN (1) < WARNING (2) < DANGER (3) < CRITICAL (4)
        priority = {"SAFE": 0, "UNCERTAIN": 1, "WARNING": 2, "DANGER": 3, "CRITICAL": 4}
        p_raw = priority[raw_zone]
        p_conf = priority[current_confirmed]

        # 2. Check for confirmation (temporal verification)
        if raw_zone == current_confirmed:
            # Already confirmed, reset counters
            w_state["consecutive_critical"] = 0
            w_state["consecutive_danger"] = 0
            w_state["consecutive_warning"] = 0
        elif p_raw > p_conf:
            # Upgrade requested: requires consecutive frame confirmations
            if raw_zone == "CRITICAL":
                w_state["consecutive_critical"] += 1
                if w_state["consecutive_critical"] >= self.critical_confirm_frames:
                    w_state["confirmed_state"] = "CRITICAL"
                    self._reset_counters(w_state)
            elif raw_zone == "DANGER":
                w_state["consecutive_danger"] += 1
                if w_state["consecutive_danger"] >= self.danger_confirm_frames:
                    w_state["confirmed_state"] = "DANGER"
                    self._reset_counters(w_state)
            elif raw_zone == "WARNING":
                w_state["consecutive_warning"] += 1
                if w_state["consecutive_warning"] >= self.warning_confirm_frames:
                    w_state["confirmed_state"] = "WARNING"
                    self._reset_counters(w_state)
            else:
                # SAFE -> UNCERTAIN transition (immediate upgrade, no verification needed)
                w_state["confirmed_state"] = raw_zone
                self._reset_counters(w_state)
        else:
            # Downgrade: Immediate transition to lower alert state (immediate cancellation)
            w_state["confirmed_state"] = raw_zone
            self._reset_counters(w_state)

        # 3. Retrieve confirmed attributes matching the output of safety_manager.evaluate_worker
        confirmed_zone = w_state["confirmed_state"]
        
        # BGR colors matching the design spec
        colors = {
            "SAFE": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "DANGER": (0, 165, 255),
            "CRITICAL": (0, 0, 255),
            "UNCERTAIN": (128, 128, 128)
        }
        color = colors.get(confirmed_zone, (0, 255, 0))

        # Generate alert attributes
        flash = (confirmed_zone in ["DANGER", "CRITICAL"])
        sound_request = "high" if confirmed_zone == "CRITICAL" else ("low" if confirmed_zone == "DANGER" else None)
        log_request = (confirmed_zone == "CRITICAL")

        return {
            "zone": confirmed_zone,
            "color": color,
            "flash": flash,
            "sound_request": sound_request,
            "log_request": log_request
        }

    def _reset_counters(self, w_state):
        w_state["consecutive_critical"] = 0
        w_state["consecutive_danger"] = 0
        w_state["consecutive_warning"] = 0

    def evaluate_global_state(self, worker_zones):
        """
        Determine the maximum confirmed danger state among all visible worker zones.
        
        Args:
            worker_zones (list): List of confirmed zone strings (e.g. ['WARNING', 'SAFE']).
            
        Returns:
            str: Global system state ('SAFE', 'UNCERTAIN', 'WARNING', 'DANGER', 'CRITICAL').
        """
        if not worker_zones:
            return "SAFE"

        priority = {"SAFE": 0, "UNCERTAIN": 1, "WARNING": 2, "DANGER": 3, "CRITICAL": 4}
        reverse_priority = {0: "SAFE", 1: "UNCERTAIN", 2: "WARNING", 3: "DANGER", 4: "CRITICAL"}

        max_pri = 0
        for zone in worker_zones:
            pri = priority.get(zone, 0)
            if pri > max_pri:
                max_pri = pri

        return reverse_priority[max_pri]
