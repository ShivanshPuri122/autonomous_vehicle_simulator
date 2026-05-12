from ultralytics import YOLO
import os

def run_inference():
    base_dir = r"d:\Projects\autonomous_vehicle_simulator"
    model_path = os.path.join(base_dir, "models", "lane_detection", "weights", "best.pt")
    source_path = os.path.join(base_dir, "data", "test_video", "test_video_2.mp4")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    if not os.path.exists(source_path):
        print(f"Test video not found at {source_path}")
        return
        
    print(f"Running Inference for Lane Detection Model: {model_path}")
    model = YOLO(model_path)
    
    # Run prediction
    results = model.predict(source=source_path, save=True, project=os.path.join(base_dir, 'test_results'), name='lane_detection_inference')
    print("Inference completed.")

if __name__ == "__main__":
    run_inference()
