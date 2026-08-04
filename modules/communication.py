import socket
import json
import logging

logger = logging.getLogger(__name__)

class WearableCommunicator:
    def __init__(self, config=None):
        """
        Args:
            config (dict): Configurations containing:
                - base_ip (str): Starting IP address, e.g. '192.168.1.100'
                - udp_host (str): loopback host to send simulated UDP packets to, default '127.0.0.1'
                - udp_port (int): loopback port, default 5005
        """
        if config is None:
            config = {}
            
        self.base_ip = config.get("base_ip", "192.168.1.100")
        self.udp_host = config.get("udp_host", "127.0.0.1")
        self.udp_port = config.get("udp_port", 5005)
        
        # Initialize UDP socket for loopback simulation
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"WearableCommunicator initialized. Base IP={self.base_ip}, "
                    f"UDP Target={self.udp_host}:{self.udp_port}")

    def get_device_ip(self, worker_id):
        """
        Maps worker tracker ID to simulated IP address dynamically.
        """
        try:
            octets = self.base_ip.split(".")
            prefix = ".".join(octets[:3])
            last_octet = int(octets[3])
            return f"{prefix}.{last_octet + worker_id}"
        except Exception as e:
            logger.warning(f"Error calculating IP for worker ID {worker_id}: {e}. Falling back to base IP.")
            return self.base_ip

    def send_alert(self, worker_id, zone):
        """
        Simulates sending warning alert signals to the worker's wearable device.
        Actual JSON commands are sent via local loopback UDP socket.
        
        Args:
            worker_id (int): Persistent tracker ID.
            zone (str): Threat zone ('SAFE', 'WARNING', 'DANGER', 'CRITICAL').
            
        Returns:
            dict: The simulated alert packet payload.
        """
        device_ip = self.get_device_ip(worker_id)
        
        # Map safety zone status to device actions
        if zone == "WARNING":
            vibration = "single_pulse"
            light = "off"
            sound = "off"
        elif zone == "DANGER":
            vibration = "pulsed"
            light = "flashing"
            sound = "off"
        elif zone == "CRITICAL":
            vibration = "continuous_high"
            light = "flashing_red"
            sound = "on"
        else:  # SAFE
            vibration = "off"
            light = "off"
            sound = "off"

        payload = {
            "worker_id": worker_id,
            "device_ip": device_ip,
            "zone": zone,
            "vibration": vibration,
            "light": light,
            "sound": sound
        }

        # Transmit UDP message locally if not SAFE
        if zone != "SAFE":
            try:
                msg = json.dumps(payload).encode("utf-8")
                self.sock.sendto(msg, (self.udp_host, self.udp_port))
                logger.debug(f"Simulated UDP Alert Sent -> {device_ip}: {payload}")
            except Exception as e:
                # Capture transmission failure cleanly without stopping the pipeline
                logger.debug(f"Failed to transmit simulated UDP packet to {self.udp_host}:{self.udp_port}: {e}")

        return payload
