import json
from qiskit_ibm_runtime import QiskitRuntimeService

class QuantumServiceManager:
    """IBM Quantum Service Manager"""
    
    def __init__(self, config_file="config/quantum_config.json"):
        """
        Initialize quantum service manager
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.service = None
        self.backend = None
        
    def _load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_file} not found")
            raise FileNotFoundError(f"Configuration file {config_file} is required")
        except json.JSONDecodeError:
            print(f"Invalid JSON in {config_file}")
            raise ValueError(f"Invalid JSON format in {config_file}")
    
    def connect(self):
        """Connect to IBM Quantum service"""
        try:
            config = self.config["ibm_quantum"]
            self.service = QiskitRuntimeService(
                token=config["token"],
                channel=config["channel"],
                instance=config["instance"]
            )
            return True
        except Exception as e:
            print(f"Failed to connect to IBM Quantum service: {e}")
            return False
    
    def select_backend(self):
        """
        Select backend device
        
        Returns:
            Selected backend device or None
        """
        if not self.service:
            return None
        
        backend_name = self.config["ibm_quantum"]["backend"]
        
        try:
            self.backend = self.service.backend(backend_name)
            return self.backend
        except Exception as e:
            print(f"Failed to select backend {backend_name}: {e}")
            return None 