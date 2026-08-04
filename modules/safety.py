import logging

logger = logging.getLogger(__name__)

class SafetyZoneManager:
    def __init__(self, safety_config=None):
        """
        Args:
            safety_config (dict): Configuration containing safety thresholds:
                - warning_threshold (float)
                - danger_threshold (float)
                - critical_threshold (float)
        """
        if safety_config is None:
            safety_config = {}
        
        self.warning_threshold = safety_config.get("warning_threshold", 100.0)
        self.danger_threshold = safety_config.get("danger_threshold", 150.0)
        self.critical_threshold = safety_config.get("critical_threshold", 200.0)
        
        logger.info(f"SafetyZoneManager initialized with thresholds: "
                    f"Warning={self.warning_threshold}, Danger={self.danger_threshold}, Critical={self.critical_threshold}")

    def evaluate_worker(self, worker):
        """
        Evaluates the safety zone of a single worker based on relative depth.
        
        Returns:
            dict: Zone info containing:
                - zone (str): 'SAFE', 'WARNING', 'DANGER', 'CRITICAL'
                - color (tuple): BGR color code for overlays
                - flash (bool): True if overlay should flash
                - sound_request (str): None, 'low', or 'high'
                - log_request (bool): True if collision logging is required
        """
        depth = worker.get("relative_depth", 0.0)
        
        # BGR Colors
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        RED = (0, 0, 255)
        
        if depth < self.warning_threshold:
            return {
                "zone": "SAFE",
                "color": GREEN,
                "flash": False,
                "sound_request": None,
                "log_request": False
            }
        elif depth < self.danger_threshold:
            return {
                "zone": "WARNING",
                "color": YELLOW,
                "flash": False,
                "sound_request": None,
                "log_request": False
            }
        elif depth < self.critical_threshold:
            return {
                "zone": "DANGER",
                "color": RED,
                "flash": True,
                "sound_request": "low",
                "log_request": False
            }
        else:
            return {
                "zone": "CRITICAL",
                "color": RED,
                "flash": True,
                "sound_request": "high",
                "log_request": True
            }

    def evaluate_system_state(self, workers):
        """
        Evaluates safety zones for all workers and determines overall system threat level
        and aggregate alert actions.
        
        Args:
            workers (list): List of workers sorted by proximity (closest first).
            
        Returns:
            tuple: (system_state, system_actions)
                - system_state (str): 'SAFE', 'WARNING', 'DANGER', 'CRITICAL'
                - system_actions (dict):
                    - border_flash (bool)
                    - fullscreen_flash (bool)
                    - sound_request (str): None, 'low', or 'high'
                    - log_request (bool)
                    - closest_worker_id (int or None)
        """
        # If no workers, system is Safe
        if not workers:
            return "SAFE", {
                "border_flash": False,
                "fullscreen_flash": False,
                "sound_request": None,
                "log_request": False,
                "closest_worker_id": None
            }

        # Initialize aggregate states
        max_zone = "SAFE"
        border_flash = False
        fullscreen_flash = False
        sound_request = None
        log_request = False
        
        # Priority mapping for max zone resolution
        zone_priority = {"SAFE": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}
        reverse_priority = {0: "SAFE", 1: "WARNING", 2: "DANGER", 3: "CRITICAL"}
        
        max_priority = 0
        
        for worker in workers:
            zone_info = self.evaluate_worker(worker)
            zone = zone_info["zone"]
            curr_priority = zone_priority[zone]
            
            if curr_priority > max_priority:
                max_priority = curr_priority
                
            if zone == "DANGER":
                border_flash = True
                if sound_request != "high":
                    sound_request = "low"
            elif zone == "CRITICAL":
                border_flash = True
                fullscreen_flash = True
                sound_request = "high"
                log_request = True
                
        system_state = reverse_priority[max_priority]
        closest_worker_id = workers[0]["id"]
        
        return system_state, {
            "border_flash": border_flash,
            "fullscreen_flash": fullscreen_flash,
            "sound_request": sound_request,
            "log_request": log_request,
            "closest_worker_id": closest_worker_id
        }
