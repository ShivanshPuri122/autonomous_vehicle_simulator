import os
import sys
import time
import cv2
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.carla_client import CarlaClient
from sensors.sensor_manager import SensorManager

def main():
    # Paths
    config_path = os.path.join('config', 'carla_settings.yaml')
    
    # 1. Initialize CARLA Client
    sim_client = CarlaClient(config_path)
    
    try:
        # 2. Spawn Vehicle
        vehicle = sim_client.spawn_vehicle()
        vehicle.set_autopilot(True) # Enable CARLA autopilot for testing
        
        # 3. Initialize Sensor Manager
        sensor_manager = SensorManager(vehicle, config_path)
        
        print("Simulation running. Press Ctrl+C to stop.")
        
        while True:
            # 4. Get Live Sensor Data
            data = sensor_manager.get_all_data()
            
            # Process Camera Data (RGB)
            image = data.get('camera')
            if image is not None:
                # Here is where you would call:
                # from perception.lane_detection.video_inference import process_frame
                # annotated_frame, stats, offset = process_frame(model, image)
                
                # For now, we display the raw feed to verify integration
                cv2.imshow('CARLA Live Camera', image)
            
            # Process LiDAR Data
            lidar = data.get('lidar')
            if lidar is not None:
                # LiDAR processing (e.g., object detection integration)
                pass

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # Wait for a bit to match simulation frequency
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        # Cleanup
        if 'sensor_manager' in locals():
            sensor_manager.destroy_all()
        if 'sim_client' in locals():
            sim_client.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
