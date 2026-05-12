import carla
import yaml
import random

class CarlaClient:
    """
    Handles connection to CARLA server and vehicle management.
    Loads environment settings from config/carla_settings.yaml.
    """
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.client = None
        self.world = None
        self.vehicle = None
        
        self._connect()

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _connect(self):
        """Connects to the CARLA server and sets up the world."""
        carla_cfg = self.config.get('carla', {})
        self.client = carla.Client(
            carla_cfg.get('host', 'localhost'), 
            carla_cfg.get('port', 2000)
        )
        self.client.set_timeout(carla_cfg.get('timeout', 10.0))
        
        # Load town and weather
        self.world = self.client.get_world()
        town = carla_cfg.get('town', 'Town04')
        if self.world.get_map().name.split('/')[-1] != town:
            self.world = self.client.load_world(town)
            
        weather = getattr(carla.WeatherParameters, carla_cfg.get('weather', 'ClearNoon'))
        self.world.set_weather(weather)
        print(f"Connected to CARLA: {town} with {carla_cfg.get('weather')} weather.")

    def spawn_vehicle(self):
        """Spawns the ego vehicle at a predefined spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        veh_cfg = self.config.get('vehicle', {})
        
        bp = blueprint_library.find(veh_cfg.get('blueprint', 'vehicle.tesla.model3'))
        bp.set_attribute('role_name', 'ego_vehicle')
        
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_idx = veh_cfg.get('spawn_index', 0)
        
        if spawn_idx < len(spawn_points):
            spawn_point = spawn_points[spawn_idx]
        else:
            spawn_point = random.choice(spawn_points)
            
        self.vehicle = self.world.spawn_actor(bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        return self.vehicle

    def get_world(self):
        return self.world

    def destroy(self):
        """Cleanup simulation actors."""
        if self.vehicle:
            self.vehicle.destroy()
            print("Vehicle destroyed.")
