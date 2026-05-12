import carla
import numpy as np

class LidarSensor:
    """
    Handles LiDAR sensor setup and data processing.
    Captures point cloud data from the simulation.
    """
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Load config parameters
        self.config = config.get('lidar', {})
        self.channels = self.config.get('channels', 32)
        self.range = self.config.get('range', 50)
        self.points_per_second = self.config.get('points_per_second', 100000)
        self.rotation_frequency = self.config.get('rotation_frequency', 20)
        
        self.sensor = None
        self.point_cloud = None
        
        self._spawn_sensor()

    def _spawn_sensor(self):
        """Spawns the LiDAR sensor and attaches it to the vehicle."""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', str(self.channels))
        lidar_bp.set_attribute('range', str(self.range))
        lidar_bp.set_attribute('points_per_second', str(self.points_per_second))
        lidar_bp.set_attribute('rotation_frequency', str(self.rotation_frequency))
        
        # Set relative position from config
        spawn_point = carla.Transform(
            carla.Location(
                x=self.config.get('x', 0.0), 
                y=self.config.get('y', 0.0), 
                z=self.config.get('z', 2.8)
            )
        )
        
        self.sensor = self.world.spawn_actor(
            lidar_bp, 
            spawn_point, 
            attach_to=self.vehicle
        )
        
        # Start listening to the sensor stream
        self.sensor.listen(lambda data: self._process_lidar(data))

    def _process_lidar(self, data):
        """Callback to process raw LiDAR data into a NumPy array."""
        # Convert raw buffer to [x, y, z, intensity] array
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self.point_cloud = points

    def get_data(self):
        """Returns the latest point cloud data."""
        return self.point_cloud

    def destroy(self):
        """Cleanup sensor actor."""
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
            print("LiDAR sensor destroyed.")
