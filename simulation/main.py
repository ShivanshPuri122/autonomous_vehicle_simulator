import os
import sys
import time
import cv2
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.carla_client import CarlaClient
from sensors.sensor_manager import SensorManager
from perception.depth_estimation import (
    estimate_obstacle_distances,
    create_depth_colormap,
    draw_distance_annotations,
)

def main():
    # Paths
    config_path = os.path.join('config', 'carla_settings.yaml')
    
    # Optional: Load YOLO model for live object detection
    obj_model = None
    obj_model_path = os.path.join('models', 'object_detection', 'weights', 'best.pt')
    if os.path.exists(obj_model_path):
        try:
            from ultralytics import YOLO
            obj_model = YOLO(obj_model_path)
            print(f"Loaded object detection model: {obj_model_path}")
        except ImportError:
            print("ultralytics not installed — running without live object detection.")
    
    # 1. Initialize CARLA Client
    sim_client = CarlaClient(config_path)
    
    try:
        # 2. Spawn Vehicle
        vehicle = sim_client.spawn_vehicle()
        vehicle.set_autopilot(True) # Enable CARLA autopilot for testing
        
        # 3. Initialize Sensor Manager (now includes depth camera)
        sensor_manager = SensorManager(vehicle, config_path)
        
        print("Simulation running. Press 'q' to stop.")
        
        while True:
            # 4. Get Live Sensor Data
            data = sensor_manager.get_all_data()
            
            # ── Process Camera Data (RGB) ────────────────────────────────
            image = data.get('camera')
            depth_map = data.get('depth')
            
            if image is not None:
                annotated = image.copy()
                
                # ── Object Detection + Depth Fusion ──────────────────────
                if obj_model is not None and depth_map is not None:
                    results = obj_model.predict(
                        source=image, conf=0.4, verbose=False
                    )[0]
                    
                    # Convert YOLO results to detection dicts
                    detections = []
                    if results.boxes is not None:
                        for box in results.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'label': results.names[int(box.cls[0])],
                                'confidence': float(box.conf[0]),
                            })
                    
                    # Enrich with depth distances
                    detections = estimate_obstacle_distances(depth_map, detections)
                    
                    # Draw annotated bounding boxes with distances
                    annotated = draw_distance_annotations(annotated, detections)
                    
                    # Build perception data for decision maker
                    perception_data = {
                        'obstacles': [
                            {
                                'label': d['label'],
                                'confidence': d['confidence'],
                                'distance': d['distance'],
                                'lateral_offset': d.get('lateral_offset', 0.0),
                            }
                            for d in detections
                        ],
                    }
                    # perception_data can now be fed to:
                    #   decision_module.update(perception_data)
                    #   fsm.evaluate(perception_data, current_speed, waypoints)
                
                cv2.imshow('CARLA Live Camera', annotated)
                
                # ── Depth Visualisation ──────────────────────────────────
                if depth_map is not None:
                    depth_vis = create_depth_colormap(depth_map, max_range=100.0)
                    if depth_vis is not None:
                        cv2.imshow('Depth Map', depth_vis)
            
            # ── Process LiDAR Data ───────────────────────────────────────
            lidar = data.get('lidar')
            if lidar is not None:
                # LiDAR processing (e.g., 3D obstacle detection)
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

