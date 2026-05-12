import yaml
from .camera_sensor import CameraSensor
from .lidar_sensor import LidarSensor

class SensorManager:
    """
    Orchestrates multiple sensors and provides a unified interface.
    Loads configurations from YAML and manages sensor lifecycles.
    """
    def __init__(self, vehicle, config_path):
        self.vehicle = vehicle
        self.config = self._load_config(config_path)
        self.sensors = {}
        
        self._init_sensors()

    def _load_config(self, path):
        """Loads sensor and vehicle settings from a YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config at {path}: {e}")
            return {}

    def _init_sensors(self):
        """Initializes and spawns all sensors defined in the config."""
        print("Initializing sensors...")
        
        # Initialize RGB Camera
        if 'camera' in self.config:
            self.sensors['camera'] = CameraSensor(self.vehicle, self.config)
        
        # Initialize LiDAR
        if 'lidar' in self.config:
            self.sensors['lidar'] = LidarSensor(self.vehicle, self.config)

    def get_all_data(self):
        """
        Retrieves data from all active sensors.
        Returns a dictionary: {'camera': np.array, 'lidar': np.array}
        """
        data = {}
        for name, sensor in self.sensors.items():
            data[name] = sensor.get_data()
        return data

    def destroy_all(self):
        """Cleans up all spawned sensors."""
        print("Destroying all sensors...")
        for name, sensor in self.sensors.items():
            sensor.destroy()
        self.sensors.clear()
