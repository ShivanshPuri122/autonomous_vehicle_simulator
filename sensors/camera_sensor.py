import carla
import numpy as np

class CameraSensor:
    """
    Handles RGB Camera sensor setup and data processing.
    Converts CARLA raw image data to NumPy arrays for Perception processing.
    """
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Load config parameters
        self.config = config.get('camera', {})
        self.width = self.config.get('width', 640)
        self.height = self.config.get('height', 480)
        self.fov = self.config.get('fov', 90)
        
        self.sensor = None
        self.image_data = None
        
        self._spawn_sensor()

    def _spawn_sensor(self):
        """Spawns the RGB camera and attaches it to the vehicle."""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', str(self.fov))
        
        # Set relative position from config
        spawn_point = carla.Transform(
            carla.Location(
                x=self.config.get('x', 1.5), 
                y=self.config.get('y', 0.0), 
                z=self.config.get('z', 2.4)
            )
        )
        
        self.sensor = self.world.spawn_actor(
            camera_bp, 
            spawn_point, 
            attach_to=self.vehicle
        )
        
        # Start listening to the sensor stream
        self.sensor.listen(lambda image: self._process_image(image))

    def _process_image(self, image):
        """Callback to process raw CARLA image data into a NumPy BGR array."""
        # Convert raw BGRA data to BGR NumPy array
        raw_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        rgba_image = raw_data.reshape((self.height, self.width, 4))
        self.image_data = rgba_image[:, :, :3]  # Extract BGR

    def get_data(self):
        """Returns the latest processed frame."""
        return self.image_data

    def destroy(self):
        """Cleanup sensor actor."""
        if self.sensor:
            self.sensor.stop()
            self.sensor.destroy()
            print("Camera sensor destroyed.")
